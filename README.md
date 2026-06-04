# ⚽ WC2026 Predictor

Sistema de predicción del Mundial 2026 basado en 10.000 simulaciones Monte Carlo con 11 features: FIFA Ranking, forma reciente (FotMob/Opta), historial mundialista, valor de plantilla (Transfermarkt), disciplina, perfil de plantel, factor sede, variables ambientales, H2H histórico, confederación y experiencia.

---

## Setup inicial (10 minutos)

### 1. Clonar y configurar

```bash
git clone https://github.com/TU_USUARIO/wc2026.git
cd wc2026
pip install -r requirements.txt
cp .env.example .env
```

Editar `.env` con tus API keys:
```
FD_API_KEY=10e1200e22fa4cc69798452e2a90d636
APISPORTS_KEY=28996d66a66fa43f5e25720d7a74b94d
```

### 2. Verificar que las APIs funcionen

```bash
python test_apis.py
```

### 3. Generar la predicción base (solo la primera vez)

```bash
python main.py
```

Genera `output/prode_completo.json` y `output/monte_carlo_results.json`.

### 4. Testear el pipeline dinámico

```bash
python src/live_updater.py --demo   # con datos de prueba
python src/live_updater.py          # con la API real (desde el 11 jun)
```

---

## GitHub Actions — Auto-update

El workflow `.github/workflows/update_predictions.yml` corre automáticamente cada 45 minutos durante el torneo.

### Configurar secrets en GitHub

1. Ir a tu repo → **Settings** → **Secrets and variables** → **Actions**
2. Crear dos secrets:

| Secret | Valor |
|---|---|
| `FD_API_KEY` | Tu key de football-data.org |
| `APISPORTS_KEY` | Tu key de api-sports.io |

3. Ir a **Settings** → **Actions** → **General** → activar **Read and write permissions**

El bot va a commitear automáticamente `output/live_predictions.json` después de cada actualización.

### Trigger manual

Ir a **Actions** → **Update WC2026 Predictions** → **Run workflow**.

---

## GitHub Pages — Predictor web shareable

1. Ir a **Settings** → **Pages**
2. Source: **Deploy from a branch**
3. Branch: `main` / Folder: `/docs`
4. Guardar

Tu predictor va a estar disponible en:
```
https://TU_USUARIO.github.io/wc2026/
```

### Actualizar la URL en docs/index.html

Reemplazar `TU_USUARIO` en la línea 8 de `docs/index.html`:
```javascript
const DATA_URL = "https://raw.githubusercontent.com/TU_USUARIO/wc2026/main/output/live_predictions.json";
```

---

## Widget iOS (Scriptable)

1. Instalar [Scriptable](https://apps.apple.com/app/scriptable/id1405459188) desde App Store
2. Crear nuevo script → pegar el contenido de `WC2026_Widget_v2.js`
3. Reemplazar `DATA_URL` en la línea del script:
```javascript
const DATA_URL = "https://raw.githubusercontent.com/TU_USUARIO/wc2026/main/output/live_predictions.json";
```
4. Agregar widget de Scriptable a la home screen

---

## Estructura del proyecto

```
wc2026/
├── .github/
│   └── workflows/
│       └── update_predictions.yml   ← Auto-update cada 45min
├── src/
│   ├── live_updater.py              ← Punto de entrada principal
│   ├── data_fetcher.py              ← APIs: football-data.org + api-sports.io
│   ├── bayesian_updater.py          ← Actualización Bayesiana del modelo
│   ├── partial_resimulator.py       ← Monte Carlo sobre partidos restantes
│   ├── monte_carlo.py               ← Simulación pre-torneo (10k runs)
│   ├── feature_engine.py            ← 11 features por equipo
│   ├── predictor.py                 ← W/D/L + probabilidades por partido
│   ├── group_simulator.py           ← Fase de grupos
│   ├── knockout_simulator.py        ← Eliminatorias
│   ├── discipline_module.py         ← Disciplina + perfil de plantel
│   ├── form_module.py               ← Forma reciente (datos verificados)
│   ├── squad_value_module.py        ← Valor de plantilla Transfermarkt
│   ├── env_module.py                ← Variables ambientales y geográficas
│   └── h2h_module.py                ← Head-to-head histórico
├── data/
│   └── raw/fixture.py               ← 48 equipos, grupos, datos base
├── output/
│   ├── prode_completo.json          ← Predicción pre-torneo completa
│   ├── monte_carlo_results.json     ← Monte Carlo original (10k sims)
│   └── live_predictions.json        ← ⭐ Se actualiza automáticamente
├── docs/
│   └── index.html                   ← Predictor web (GitHub Pages)
├── WC2026_Widget_v2.js              ← Widget iOS (Scriptable)
├── main.py                          ← Predicción pre-torneo completa
├── test_apis.py                     ← Verificación de APIs
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Comandos útiles

```bash
# Correr todo el pipeline pre-torneo
python main.py

# Actualizar con resultados reales
python src/live_updater.py

# Ver estado actual
python src/live_updater.py --status

# Test sin API
python src/live_updater.py --demo

# Verificar APIs
python test_apis.py
```

---

## Fuentes de datos

| Dato | Fuente | Confiabilidad |
|---|---|---|
| Resultados en vivo | football-data.org (API) | ✅ Alta |
| Tarjetas y lesiones | api-sports.io (API) | ✅ Alta |
| Forma reciente | FotMob/Opta + ESPN + Wikipedia | ✅ Alta |
| FIFA Rankings | FIFA.com | ✅ Alta |
| Valor de plantillas | Transfermarkt | ✅ Media-alta |
| Variables ambientales | Climate Central | ✅ Alta |
| H2H histórico | Wikipedia + ESPN | ✅ Media-alta |

---

## Métricas de accuracy (post-torneo)

Una vez terminado el Mundial (19 jul 2026), se calcularán:
- **Accuracy** por fase (grupos vs eliminatorias)
- **Brier Score** de probabilidades
- **True Positives / False Positives** por resultado
- **Upsets detectados** (sorpresas que el modelo anticipó)
