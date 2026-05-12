# ===========================================================================
# CONFIG — Tập trung toàn bộ cấu hình, không hardcode ở bất kỳ đâu khác.
# ===========================================================================
# Mọi giá trị đều có thể ghi đè qua biến môi trường (Environment Variables).
# ===========================================================================

import os
from dotenv import load_dotenv

# Nạp biến từ file .env vào os.environ
load_dotenv()

# ── Model ──
IMG_SIZE: int = int(os.getenv("IMG_SIZE", "128"))
EMBED_DIM: int = int(os.getenv("EMBED_DIM", "128"))

# ── Paths ──
MODEL_PATH: str = os.getenv("MODEL_PATH", "best_model.pth")

# ── Firebase ──
FIREBASE_CREDENTIALS_PATH: str = os.getenv("FIREBASE_CREDENTIALS_PATH", "firebase_credentials.json")

# ── ImgBB ──
IMGBB_API_KEY: str = os.getenv("IMGBB_API_KEY", "1f8c70878901f52254c899519747f80f")

# ── Verification ──
THRESHOLD: float = float(os.getenv("THRESHOLD", "0.3158"))
