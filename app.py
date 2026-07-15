import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os

# ====================== PAGE CONFIG ======================
st.set_page_config(page_title="CropVision AI - Hybrid NIR+VARI", layout="wide")
st.title("🌾 CropVision AI")
st.markdown("**Hybrid plant health analysis** – RGB to NIR prediction + VARI fusion, with automatic domain detection (aerial/ground).")

# ====================== MODEL DEFINITION ======================
class EfficientUNetGenerator(nn.Module):
    def __init__(self, base_channels=40):
        super().__init__()
        bc = base_channels
        self.enc1 = self._down_block(3, bc)
        self.enc2 = self._down_block(bc, bc*2)
        self.enc3 = self._down_block(bc*2, bc*4)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(bc*4, bc*4, 3, padding=1, bias=False),
            nn.BatchNorm2d(bc*4), nn.ReLU(inplace=True),
            nn.Conv2d(bc*4, bc*4, 3, padding=1, bias=False),
            nn.BatchNorm2d(bc*4), nn.ReLU(inplace=True),
        )
        self.up3  = nn.ConvTranspose2d(bc*4, bc*2, 4, 2, 1)
        self.dec3 = self._up_block(bc*4, bc*2)
        self.up2  = nn.ConvTranspose2d(bc*2, bc, 4, 2, 1)
        self.dec2 = self._up_block(bc*2, bc)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(bc, bc//2, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(bc//2, 1, 3, padding=1),
            nn.Tanh()
        )

    def _down_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.LeakyReLU(0.2, inplace=True),
        )

    def _up_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        b  = self.bottleneck(e3)
        d3 = self.dec3(torch.cat([self.up3(b), e2], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e1], 1))
        return self.final(d2)

# ====================== LOAD MODEL (CACHED) ======================
@st.cache_resource
def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EfficientUNetGenerator(base_channels=40).to(device)
    if not os.path.exists(model_path):
        st.error(f"Model not found at `{model_path}`. Please place your trained `.pth` file there or adjust the path.")
        st.stop()
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, device

# ====================== TILED PREDICTION ======================
def tiled_predict(rgb_np, model, device, tile_size=256, overlap=128):
    h, w = rgb_np.shape[:2]
    nir_acc    = np.zeros((h, w), dtype=np.float64)
    weight_acc = np.zeros((h, w), dtype=np.float64)
    step       = tile_size - overlap

    k   = cv2.getGaussianKernel(tile_size, tile_size / 5)
    win = (k @ k.T).astype(np.float64)
    win /= win.max()

    tile_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    ys = sorted(set(list(range(0, max(1, h - tile_size), step)) + [max(0, h - tile_size)]))
    xs = sorted(set(list(range(0, max(1, w - tile_size), step)) + [max(0, w - tile_size)]))

    with torch.no_grad():
        for y in ys:
            for x in xs:
                y2, x2 = min(y + tile_size, h), min(x + tile_size, w)
                tile   = rgb_np[y:y2, x:x2]
                th, tw = tile.shape[:2]
                if th < 16 or tw < 16:
                    continue
                padded = np.zeros((tile_size, tile_size, 3), dtype=rgb_np.dtype)
                padded[:th, :tw] = tile
                inp = tile_tf(Image.fromarray(padded)).unsqueeze(0).to(device)
                pred_nir = (model(inp) * 0.5 + 0.5).clamp(0, 1).squeeze().cpu().numpy()
                nir_acc[y:y2, x:x2]    += pred_nir[:th, :tw] * win[:th, :tw]
                weight_acc[y:y2, x:x2] += win[:th, :tw]

    weight_acc = np.where(weight_acc < 1e-8, 1e-8, weight_acc)
    result = (nir_acc / weight_acc).astype(np.float32)
    result_u8 = cv2.bilateralFilter(
        (result * 255).clip(0, 255).astype(np.uint8),
        d=9, sigmaColor=25, sigmaSpace=25
    )
    return result_u8.astype(np.float32) / 255.0

# ====================== DOMAIN DETECTION ======================
def detect_domain(rgb_np):
    gray = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2GRAY).astype(np.float32)
    h, w = gray.shape

    local_var = cv2.Laplacian(gray, cv2.CV_32F).var()
    var_score = np.clip(local_var / 1200.0, 0, 1)

    fft = np.abs(np.fft.fft2(gray))
    fft_shift = np.fft.fftshift(fft)
    center_r = h // 2
    horiz_band = fft_shift[center_r-5:center_r+5, :]
    horiz_energy = horiz_band.sum() / (fft_shift.sum() + 1e-6)
    row_score = 1.0 - np.clip(horiz_energy * 20, 0, 1)

    edges = cv2.Canny(gray.astype(np.uint8), 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80,
                            minLineLength=w//6, maxLineGap=20)
    line_score = 0.0
    if lines is not None:
        line_score = np.clip(len(lines) / 30.0, 0, 1)

    hsv = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2HSV)
    hue_std = hsv[:,:,0].std()
    hue_score = np.clip(hue_std / 40.0, 0, 1)

    ground_probability = (0.30 * var_score + 0.25 * row_score +
                          0.25 * line_score + 0.20 * hue_score)
    domain = 'ground' if ground_probability > 0.5 else 'aerial'
    return domain, float(ground_probability)

# ====================== RGB INDICES ======================
def compute_rgb_indices(rgb_np):
    img_f = rgb_np.astype(np.float32) / 255.0
    R, G, B = img_f[:,:,0], img_f[:,:,1], img_f[:,:,2]
    VARI = np.clip((G - R) / (G + R - B + 1e-6), -1.0, 1.0)
    return VARI

# ====================== NON-VEGETATION MASK ======================
def get_non_veg_mask(rgb_np):
    img_f = rgb_np.astype(np.float32) / 255.0
    R, G, B = img_f[:,:,0], img_f[:,:,1], img_f[:,:,2]
    ExG = 2*G - R - B
    hsv = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2HSV).astype(np.float32)
    hue = hsv[:,:,0]
    sat = hsv[:,:,1] / 255.0
    val = hsv[:,:,2] / 255.0

    is_road   = (sat < 0.15) & (val > 0.25)
    is_shadow = val < 0.12
    is_soil   = ((hue > 5) & (hue < 32)) & (ExG < 0.0) & (sat > 0.1)
    is_blue   = (hue > 100) & (hue < 140) & (sat > 0.3)

    non_veg = is_road | is_shadow | is_soil | is_blue
    non_veg = cv2.morphologyEx(
        non_veg.astype(np.uint8), cv2.MORPH_CLOSE,
        np.ones((7, 7), np.uint8)
    ).astype(bool)
    return non_veg

# ====================== FUSION ======================
def fuse_indices(model_ndvi, vari, domain, ground_prob):
    if domain == 'aerial':
        w_model = 0.80
        w_vari = 0.20
    else:
        w_model = 0.80 - 0.65 * ((ground_prob - 0.5) / 0.5)
        w_model = np.clip(w_model, 0.15, 0.80)
        w_vari = 1.0 - w_model
    fused = w_model * model_ndvi + w_vari * vari
    fused = np.clip(fused, -1.0, 1.0)
    return fused, w_model, w_vari

# ====================== MAIN ANALYSIS FUNCTION ======================
def hybrid_analyze(rgb_np, model, device, force_domain=None):
    h, w = rgb_np.shape[:2]

    # Domain detection
    if force_domain is None:
        domain, ground_prob = detect_domain(rgb_np)
    else:
        domain = force_domain
        ground_prob = 1.0 if force_domain == 'ground' else 0.0

    # Model NIR → NDVI
    pred_nir = tiled_predict(rgb_np, model, device, tile_size=256, overlap=128)
    if pred_nir.shape != (h, w):
        pred_nir = cv2.resize(pred_nir, (w, h), interpolation=cv2.INTER_LINEAR)

    red = rgb_np[:, :, 0].astype(np.float32) / 255.0
    model_ndvi = np.clip((pred_nir - red) / (pred_nir + red + 1e-6), -1.0, 1.0)

    # RGB indices
    vari = compute_rgb_indices(rgb_np)

    # Fusion
    fused_index, w_model, w_vari = fuse_indices(model_ndvi, vari, domain, ground_prob)

    # Non-veg mask
    non_veg = get_non_veg_mask(rgb_np)

    # Plant mask
    plant_mask = (fused_index > 0.1) & (~non_veg)
    plant_mask = cv2.morphologyEx(plant_mask.astype(np.uint8)*255, cv2.MORPH_OPEN, np.ones((4,4), np.uint8))
    plant_mask = cv2.morphologyEx(plant_mask, cv2.MORPH_CLOSE, np.ones((9,9), np.uint8))
    plant_mask = plant_mask > 128

    # Statistics
    valid = fused_index[~non_veg]
    plant_vals = fused_index[plant_mask]
    field_mean = float(np.mean(valid)) if len(valid) > 0 else 0.0
    plant_mean = float(np.mean(plant_vals)) if len(plant_vals) > 0 else 0.0
    coverage = np.sum(plant_mask) / (~non_veg).sum() * 100

    h_pct = np.sum(plant_mask & (fused_index > 0.35)) / (np.sum(plant_mask)+1e-6) * 100
    m_pct = np.sum(plant_mask & (fused_index > 0.15) & (fused_index <= 0.35)) / (np.sum(plant_mask)+1e-6) * 100
    s_pct = np.sum(plant_mask & (fused_index >= 0.0) & (fused_index <= 0.15)) / (np.sum(plant_mask)+1e-6) * 100
    sv_pct = np.sum(plant_mask & (fused_index < 0.0)) / (np.sum(plant_mask)+1e-6) * 100

    # Health colour overlay
    health_map = np.zeros_like(rgb_np)
    health_map[plant_mask & (fused_index > 0.35)] = [0, 180, 0]
    health_map[plant_mask & (fused_index > 0.15) & (fused_index <= 0.35)] = [180, 220, 0]
    health_map[plant_mask & (fused_index >= 0.0) & (fused_index <= 0.15)] = [255, 200, 0]
    health_map[plant_mask & (fused_index < 0.0)] = [200, 50, 0]

    health_vis = rgb_np.copy().astype(np.float32)
    health_vis[plant_mask] = rgb_np[plant_mask] * 0.35 + health_map[plant_mask] * 0.65
    health_vis = np.clip(health_vis, 0, 255).astype(np.uint8)

    # Plant mask overlay for display
    overlay = rgb_np.copy().astype(np.float32)
    gt = np.zeros_like(rgb_np); gt[:,:,1] = 160
    overlay[plant_mask] = rgb_np[plant_mask]*0.5 + gt[plant_mask]*0.5
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    return {
        "rgb": rgb_np,
        "model_ndvi": model_ndvi,
        "vari": vari,
        "fused_index": fused_index,
        "plant_mask": plant_mask,
        "overlay": overlay,
        "health_vis": health_vis,
        "domain": domain,
        "ground_prob": ground_prob,
        "w_model": w_model,
        "w_vari": w_vari,
        "field_mean": field_mean,
        "plant_mean": plant_mean,
        "coverage": coverage,
        "healthy_pct": h_pct,
        "moderate_pct": m_pct,
        "stressed_pct": s_pct,
        "severe_pct": sv_pct,
    }

# ====================== STREAMLIT UI ======================
def main():
    # Sidebar settings
    st.sidebar.header("Model & Settings")
    model_path = st.sidebar.text_input("Path to model (.pth)", value="models/finetuned_epoch40.pth")
    force_domain = st.sidebar.radio("Force domain (optional)", ["Auto", "Aerial", "Ground"])
    force_val = None if force_domain == "Auto" else force_domain.lower()

    # Load model
    if not os.path.exists(model_path):
        st.sidebar.error(f"Model not found at `{model_path}`. Please upload or correct path.")
        st.stop()
    model, device = load_model(model_path)
    st.sidebar.success(f"Model loaded on {device}")

    # Upload image
    uploaded_file = st.file_uploader("Upload a crop image (RGB)", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        # Read image
        pil_img = Image.open(uploaded_file).convert("RGB")
        rgb_np = np.array(pil_img)

        # Run analysis
        with st.spinner("Running hybrid analysis (tiled NIR prediction, domain detection, fusion)..."):
            res = hybrid_analyze(rgb_np, model, device, force_domain=force_val)

        # Display results
        col1, col2 = st.columns(2)
        with col1:
            st.image(res["rgb"], caption="Input RGB", use_container_width=True)
        with col2:
            st.image(res["overlay"], caption=f"Plant mask (coverage: {res['coverage']:.1f}%)", use_container_width=True)

        # Metrics row
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Domain", f"{res['domain'].upper()}", f"ground prob {res['ground_prob']:.0%}")
        m2.metric("Fusion weights", f"Model {res['w_model']:.0%} / VARI {res['w_vari']:.0%}")
        m3.metric("Field mean NDVI", f"{res['field_mean']:.3f}")
        m4.metric("Plant mean NDVI", f"{res['plant_mean']:.3f}")

        # Health distribution bars
        st.subheader("🌱 Plant health distribution")
        col_pct1, col_pct2, col_pct3, col_pct4 = st.columns(4)
        col_pct1.metric("Healthy", f"{res['healthy_pct']:.1f}%")
        col_pct2.metric("Moderate", f"{res['moderate_pct']:.1f}%")
        col_pct3.metric("Stressed", f"{res['stressed_pct']:.1f}%")
        col_pct4.metric("Severe", f"{res['severe_pct']:.1f}%")
        st.progress(res['healthy_pct']/100, text="Healthy")
        st.progress(res['moderate_pct']/100, text="Moderate")
        st.progress(res['stressed_pct']/100, text="Stressed")
        st.progress(res['severe_pct']/100, text="Severe")

        # Visual comparison: Model NDVI, VARI, fused, health map
        st.subheader("📊 Index maps")
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        im1 = axes[0].imshow(res["model_ndvi"], cmap='RdYlGn', vmin=-0.2, vmax=0.8)
        axes[0].set_title(f"Model NDVI (weight {res['w_model']:.0%})")
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0], fraction=0.046)
        im2 = axes[1].imshow(res["vari"], cmap='RdYlGn', vmin=-0.5, vmax=0.6)
        axes[1].set_title(f"VARI (weight {res['w_vari']:.0%})")
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1], fraction=0.046)
        im3 = axes[2].imshow(res["fused_index"], cmap='RdYlGn', vmin=-0.2, vmax=0.8)
        axes[2].set_title(f"Fused index (field mean {res['field_mean']:.3f})")
        axes[2].axis('off')
        plt.colorbar(im3, ax=axes[2], fraction=0.046)
        axes[3].imshow(res["health_vis"])
        axes[3].set_title(f"Health map (plant mean {res['plant_mean']:.3f})")
        axes[3].axis('off')
        legend_elements = [
            Patch(facecolor='#00b400', label=f'Healthy   {res["healthy_pct"]:.0f}%'),
            Patch(facecolor='#b4dc00', label=f'Moderate  {res["moderate_pct"]:.0f}%'),
            Patch(facecolor='#ffc800', label=f'Stressed  {res["stressed_pct"]:.0f}%'),
            Patch(facecolor='#c83200', label=f'Severe    {res["severe_pct"]:.0f}%'),
        ]
        axes[3].legend(handles=legend_elements, loc='lower left', fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)

        # Option to download results (optional)
        st.download_button(
            label="Download health map (PNG)",
            data=cv2.imencode('.png', cv2.cvtColor(res["health_vis"], cv2.COLOR_RGB2BGR))[1].tobytes(),
            file_name="health_map.png",
            mime="image/png"
        )

if __name__ == "__main__":
    main()