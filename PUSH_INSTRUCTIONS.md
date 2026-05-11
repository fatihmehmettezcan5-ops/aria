# GitHub'a Push Etme — Adım Adım

Workspace'teki **`aria.zip`** (371 KB) dosyasını indir. İçinde tam çalışır
proje + git geçmişi (`.git/`) var.

## 1) İndir + aç

```bash
# Bilgisayarına indir, sonra:
unzip aria.zip
cd aria
git log --oneline       # tek bir initial commit göreceksin
```

## 2) GitHub'da boş repo oluştur

GitHub.com → **+** → **New repository**:
- Name: `aria` (veya istediğin)
- **Initialize seçeneklerini İŞARETLEMEDEN** (README, .gitignore, license eklemesin)
- **Create repository**

## 3) Remote ekle + push

GitHub sana üst kısımda iki URL verir. **HTTPS** (klasik) veya **SSH** (anahtar varsa).

### HTTPS ile:
```bash
git remote add origin https://github.com/KULLANICI_ADIN/aria.git
git branch -M main
git push -u origin main
```
(Şifre yerine **Personal Access Token** kullan — GitHub Settings → Developer
settings → Personal access tokens → Generate new token (classic) → repo
scope. Token'ı şifre olarak yapıştır.)

### SSH ile (önerilen):
```bash
git remote add origin git@github.com:KULLANICI_ADIN/aria.git
git branch -M main
git push -u origin main
```

## 4) Doğrula

`https://github.com/KULLANICI_ADIN/aria` aç. README otomatik render
edilecek, mimari diyagramı + komutları göreceksin.

## 5) (Opsiyonel) Smoke train + canlı dene

Repo'yu klonladığın yerde:
```bash
make setup           # Python + Node bağımlılıkları
make smoke-train     # ~3 dk CPU'da
make up              # Docker compose ile 3000 portunda UI açılır
```

## Repo İçeriği Özet

- **176 dosya tracked**, ~850 KB kaynak kod
- Tüm test paketi: `pytest -q` → 29/29 passed
- `runs/` klasörü dahil değil (eğitim sıfırdan tekrar üretilebilir)
- `.env.example` var; gerçek `.env` ve secret yok

İyi çalışmalar! 🚀
