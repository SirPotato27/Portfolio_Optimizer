# Portfolio Optimizer - Trabajo Final Renta Variable

Aplicación Streamlit para resolver el taller de portafolio óptimo long-only. Incluye la demo original de 20 activos y, en modo Yahoo Finance API, permite escribir cualquier símbolo válido de Yahoo Finance.

## Funcionalidades

- Descarga precios mensuales desde Yahoo Finance para cualquier lista de símbolos válidos escrita por el usuario.
- Permite subir Excel/CSV de precios como respaldo; si no se escriben activos, usa todas las columnas numéricas excepto el benchmark.
- Incluye datos históricos de Bloomberg 30/04/2021-30/04/2026 para demo estable.
- Calcula rendimientos logarítmicos mensuales.
- Calcula riesgo mensual/anual, risk unit, correlaciones y matriz varianza-covarianza.
- Calcula beta raw y beta ajustado contra SPX.
- Estima rendimiento esperado histórico, CAPM y ANR incluido.
- Calcula portafolio 1/N, mínima varianza y máxima razón de Sharpe.
- Grafica frontera eficiente y línea tangente/CAL.
- Calcula VaR5% anual y restricciones de VaR con activo libre de riesgo.
- Exporta resultados a Excel.

## Cómo correr localmente

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Cómo desplegar en Streamlit Community Cloud

1. Crear un repositorio en GitHub.
2. Subir estos archivos al repositorio:
   - `streamlit_app.py`
   - `requirements.txt`
   - carpeta `data/`
   - carpeta `.streamlit/` opcional.
3. Entrar a Streamlit Community Cloud.
4. Click en `Create app`.
5. Seleccionar el repo, branch y archivo principal `streamlit_app.py`.
6. Click en `Deploy`.

## Uso de Yahoo Finance

En el modo `Yahoo Finance API`, escriba los tickers separados por comas, espacios o saltos de línea. Ejemplos: `AAPL, MSFT, TSLA, META, AMD, SPY, QQQ, BRK-B, ^IXIC`. Para acciones internacionales use el sufijo de Yahoo Finance, por ejemplo `7203.T` o `ASML.AS`. El benchmark para beta/CAPM también se puede cambiar; por defecto se usa `^GSPC`, que en el modelo se muestra como `SPX`.

## Notas

La aplicación usa como tasa libre de riesgo la tasa Treasury 10Y nominal ingresada por el usuario y la convierte a efectiva anual como:

`Rf EA = (1 + tasa_nominal/2)^2 - 1`

El VaR5% se calcula como:

`VaR5% = E[r] - Z(95%) * sigma`

