"""
💪 INDICADOR FUERZA-IQ
Diseño profesional: gráfico velas (izquierda), círculo señal (centro), panel activos (derecha).
Señal en segundo 58 con IA RandomForest y entrenamiento continuo.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pytz
from datetime import datetime, timedelta
import time
import logging
import warnings
warnings.filterwarnings('ignore')

from streamlit_autorefresh import st_autorefresh
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

# --- Configuración de logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Importar API de IQ Option con manejo de error ---
try:
    from iqoptionapi.stable_api import IQ_Option
    IQ_AVAILABLE = True
except ImportError as e:
    IQ_AVAILABLE = False
    st.error("""
    ⚠️ **Error crítico:** No se pudo importar la librería `iqoptionapi`.
    Verifica que la línea en `requirements.txt` sea exactamente:
    `git+https://github.com/williansandi/iqoptionapi-2025-Atualizada-.git#egg=iqoptionapi`
    """)

# --- Configuración de la página ---
st.set_page_config(
    page_title="FUERZA-IQ",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Autorefresh CADA 1 SEGUNDO para precisión en segundo 58.
st_autorefresh(interval=1000, key="segundo_a_segundo")

# --- Zona horaria (Ecuador) ---
ecuador_tz = pytz.timezone('America/Guayaquil')

# ============================================
# 1. CLASE DE CONEXIÓN MEJORADA (CON REINTENTOS)
# ============================================
class IQOptionConnector:
    def __init__(self):
        self.api = None
        self.conectado = False
        self.balance = 0
        self.tipo_cuenta = "PRACTICE"
        self.activos_cache = {}
        self.ultima_actualizacion_activos = 0
        self.email = None
        self.password = None

    def conectar(self, email, password):
        """Conecta con reintentos (máx 3) y backoff exponencial."""
        if not IQ_AVAILABLE:
            return False, "Librería no disponible"
        self.email, self.password = email, password
        reintentos = 3
        espera = 2
        for intento in range(reintentos):
            try:
                self.api = IQ_Option(email, password)
                check, reason = self.api.connect()
                if check:
                    self.conectado = True
                    self.balance = self.api.get_balance()
                    logging.info("Conexión exitosa")
                    return True, "Conectado"
                else:
                    logging.warning(f"Intento {intento+1} falló: {reason}")
            except Exception as e:
                logging.error(f"Error en conexión: {e}")
            if intento < reintentos - 1:
                time.sleep(espera)
                espera *= 2
        self.conectado = False
        return False, "No se pudo conectar tras reintentos"

    def cambiar_balance(self, tipo="PRACTICE"):
        if self.conectado:
            try:
                self.api.change_balance(tipo)
                self.tipo_cuenta = tipo
                time.sleep(1)
                self.balance = self.api.get_balance()
                return True
            except:
                return False
        return False

    def obtener_saldo(self):
        if self.conectado:
            try:
                self.balance = self.api.get_balance()
            except:
                pass
        return self.balance

    def obtener_activos_otc(self, max_activos=100):
        """Obtiene y cachea la lista de activos OTC."""
        ahora = time.time()
        if ahora - self.ultima_actualizacion_activos < 600 and self.activos_cache:
            return self.activos_cache

        if not self.conectado:
            return []
        try:
            activos_data = self.api.get_all_open_time()
            activos = []
            for categoria in ["binary", "turbo"]:
                for activo, data in activos_data.get(categoria, {}).items():
                    if data.get("open", False) and "-OTC" in activo:
                        activos.append(activo)
            self.activos_cache = sorted(activos)[:max_activos]
            self.ultima_actualizacion_activos = ahora
            logging.info(f"Lista de activos OTC actualizada. {len(self.activos_cache)} encontrados.")
            return self.activos_cache
        except Exception as e:
            logging.error(f"Error obteniendo activos: {e}")
            return self.activos_cache if self.activos_cache else []

    def obtener_datos_vela_1min(self, activo, reintentos=2):
        """Obtiene los datos de la vela de 1 minuto más reciente, incluyendo volumen y valores."""
        if not self.conectado:
            return None
        for intento in range(reintentos):
            try:
                velas = self.api.get_candles(activo, 60, 10, time.time())
                if not velas or len(velas) < 2:
                    if intento == reintentos-1:
                        return None
                    time.sleep(2)
                    continue
                df = pd.DataFrame(velas)
                df['from'] = pd.to_datetime(df['from'], unit='s')
                df = df.set_index('from')
                df = df[['open', 'close', 'max', 'min', 'volume']].astype(float)
                df = df.rename(columns={'max': 'high', 'min': 'low'})
                df = df.sort_index()
                return df
            except Exception as e:
                logging.error(f"Error obteniendo velas de {activo}: {e}")
                if intento == reintentos-1:
                    return None
                time.sleep(2)
        return None

    def calcular_fuerza_y_volumen(self, df):
        """
        Calcula las features clave a partir del DataFrame de velas de 1 minuto.
        df: DataFrame con las últimas 10 velas.
        """
        if df is None or len(df) < 2:
            return None
        ult = df.iloc[-1]
        prev = df.iloc[-2]

        # --- Features basadas en volumen y fuerza ---
        # 1. Volumen ratio (volumen actual / media de volumen de las últimas 9)
        vol_ma = df['volume'].iloc[-9:].mean()
        volumen_ratio = ult['volume'] / vol_ma if vol_ma > 0 else 1.0

        # 2. Fuerza de compra (simulada, ya que la API no da valores monetarios por lado)
        rango = ult['high'] - ult['low']
        if ult['close'] > ult['open']:
            fuerza_compra = 50 + 50 * ( (ult['close'] - ult['open']) / rango ) if rango > 0 else 75
            dif_compra_venta = 1.0
        else:
            fuerza_compra = 50 - 50 * ( (ult['open'] - ult['close']) / rango ) if rango > 0 else 25
            dif_compra_venta = -1.0

        # 3. Momentum (cambio en el precio de cierre en 3 velas)
        if len(df) >= 3:
            precio_hace_3 = df['close'].iloc[-3]
            momentum = (ult['close'] - precio_hace_3) / precio_hace_3 * 100
        else:
            momentum = 0

        # 4. RSI (sobre 7 períodos)
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=7, min_periods=1).mean().iloc[-1]
        avg_loss = loss.rolling(window=7, min_periods=1).mean().iloc[-1]
        rs = avg_gain / avg_loss if avg_loss != 0 else 100
        rsi = 100 - (100 / (1 + rs))

        # 5. Log Volatilidad (desviación estándar de retornos)
        retornos = df['close'].pct_change().dropna()
        log_vol = np.log1p(retornos.std()) if len(retornos) > 0 else 0

        # 6. Posición en BB (simplificada)
        bb_std = df['close'].rolling(5).std().iloc[-1] * 2
        bb_sup = df['close'].rolling(5).mean().iloc[-1] + bb_std
        bb_inf = df['close'].rolling(5).mean().iloc[-1] - bb_std
        if bb_sup - bb_inf > 0:
            bb_pos = (ult['close'] - bb_inf) / (bb_sup - bb_inf)
        else:
            bb_pos = 0.5

        features = {
            'fuerza_compra': fuerza_compra,
            'volumen_ratio': volumen_ratio,
            'precio_momentum': momentum,
            'rsi_14': rsi,
            'bb_posicion': bb_pos,
            'log_volatilidad': log_vol,
            'dif_compra_venta': dif_compra_venta,
            'precio_actual': ult['close'],
            'precio_apertura': ult['open'],
            'maximo': ult['high'],
            'minimo': ult['low']
        }
        return features

# ============================================
# 2. MODELO DE IA (RANDOM FOREST + ENTRENAMIENTO CONTINUO)
# ============================================
class ModeloFuerza:
    def __init__(self):
        self.modelo = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, warm_start=True)
        self.scaler = StandardScaler()
        self.entrenado = False
        self.X_buffer = []
        self.y_buffer = []
        self.buffer_size = 10

    def predecir(self, features_dict):
        """Predice la dirección de la próxima vela (1: sube, 0: baja)."""
        if not features_dict:
            return 0.5, 0
        feature_order = ['fuerza_compra', 'volumen_ratio', 'precio_momentum', 'rsi_14',
                         'bb_posicion', 'log_volatilidad', 'dif_compra_venta']
        X = np.array([[features_dict[f] for f in feature_order]])

        if self.entrenado:
            X_scaled = self.scaler.transform(X)
            prob = self.modelo.predict_proba(X_scaled)[0][1]
        else:
            prob = 0.55

        return prob

    def entrenar_o_actualizar(self, features_dict, resultado_real):
        if not features_dict:
            return
        feature_order = ['fuerza_compra', 'volumen_ratio', 'precio_momentum', 'rsi_14',
                         'bb_posicion', 'log_volatilidad', 'dif_compra_venta']
        X_new = np.array([[features_dict[f] for f in feature_order]])
        y_new = np.array([resultado_real])

        self.X_buffer.append(X_new[0])
        self.y_buffer.append(y_new[0])

        if len(self.X_buffer) >= self.buffer_size:
            X_batch = np.array(self.X_buffer)
            y_batch = np.array(self.y_buffer)

            if not self.entrenado:
                X_scaled = self.scaler.fit_transform(X_batch)
                self.modelo.fit(X_scaled, y_batch)
                self.entrenado = True
                logging.info("Modelo entrenado por primera vez.")
            else:
                X_scaled = self.scaler.transform(X_batch)
                self.modelo.n_estimators += 10
                self.modelo.fit(X_scaled, y_batch)
                logging.info("Modelo actualizado con nuevos datos.")

            self.X_buffer = []
            self.y_buffer = []

# ============================================
# 3. INTERFAZ DE USUARIO (DISEÑO DE 3 COLUMNAS)
# ============================================
def main():
    st.markdown("<h1 style='text-align: center;'>💪 INDICADOR FUERZA-IQ</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>MERCADO OTC | SEÑAL EN SEGUNDO 58</h3>", unsafe_allow_html=True)
    st.markdown("---")

    # --- Inicializar estado de sesión ---
    if 'connector' not in st.session_state:
        st.session_state.connector = IQOptionConnector()
    if 'modelo' not in st.session_state:
        st.session_state.modelo = ModeloFuerza()
    if 'conectado' not in st.session_state:
        st.session_state.conectado = False
    if 'activo_seleccionado' not in st.session_state:
        st.session_state.activo_seleccionado = None
    if 'lista_activos' not in st.session_state:
        st.session_state.lista_activos = []
    if 'senal_actual' not in st.session_state:
        st.session_state.senal_actual = None
    if 'fuerza_actual' not in st.session_state:
        st.session_state.fuerza_actual = 0
    if 'confianza_actual' not in st.session_state:
        st.session_state.confianza_actual = 0
    if 'ultimo_features' not in st.session_state:
        st.session_state.ultimo_features = None
    if 'resultado_pendiente' not in st.session_state:
        st.session_state.resultado_pendiente = None
    if 'grafico_velas' not in st.session_state:
        st.session_state.grafico_velas = None

    # --- Barra lateral de configuración (simple) ---
    with st.sidebar:
        st.markdown("### 🔐 Acceso")
        if not st.session_state.conectado:
            email = st.text_input("Email", placeholder="tu@email.com")
            password = st.text_input("Contraseña", type="password", placeholder="••••••••")
            if st.button("🔌 Conectar a IQ Option", use_container_width=True):
                with st.spinner("Conectando..."):
                    ok, msg = st.session_state.connector.conectar(email, password)
                    if ok:
                        st.session_state.conectado = True
                        st.session_state.lista_activos = st.session_state.connector.obtener_activos_otc()
                        st.success("✅ Conectado")
                        st.rerun()  # Forzar recarga para mostrar la interfaz completa
                    else:
                        st.error(f"❌ {msg}")
        else:
            st.success("✅ Conectado")
            saldo = st.session_state.connector.obtener_saldo()
            st.metric("Saldo", f"${saldo:.2f}")
            if st.button("🚪 Desconectar"):
                st.session_state.conectado = False
                st.rerun()

        if st.session_state.conectado and st.session_state.lista_activos:
            st.markdown("### ⚙️ Activo a monitorear")
            activo_idx = st.selectbox(
                "Selecciona un activo OTC",
                range(len(st.session_state.lista_activos)),
                format_func=lambda i: st.session_state.lista_activos[i].replace("-OTC", "")
            )
            st.session_state.activo_seleccionado = st.session_state.lista_activos[activo_idx]
            st.caption("Analizando 1 activo. IA aprende continuamente.")
        elif st.session_state.conectado:
            st.warning("Cargando lista de activos...")

    # --- Verificar conexión ---
    if not st.session_state.conectado:
        st.info("👆 Conéctate a IQ Option en la barra lateral para comenzar.")
        # Mostrar previsualización
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div style="background-color: #2E2E2E; width: 250px; height: 250px; border-radius: 50%; margin: 0 auto; display: flex; flex-direction: column; justify-content: center; align-items: center; border: 4px solid #AAAAAA;">
                <span style="color: white; font-size: 32px;">💤</span>
                <span style="color: white; font-size: 20px;">Esperando...</span>
            </div>
            """, unsafe_allow_html=True)
        return

    # --- Obtener la hora actual y el segundo ---
    ahora = datetime.now(ecuador_tz)
    segundo_actual = ahora.second

    # --- Lógica de análisis y señal (solo si hay activo seleccionado) ---
    if st.session_state.activo_seleccionado:
        # Obtener datos
        df_velas = st.session_state.connector.obtener_datos_vela_1min(st.session_state.activo_seleccionado)
        features = st.session_state.connector.calcular_fuerza_y_volumen(df_velas) if df_velas is not None else None

        if features is not None:
            st.session_state.ultimo_features = features

            # --- Entrenamiento con resultado de vela anterior ---
            if st.session_state.resultado_pendiente is not None:
                res = st.session_state.resultado_pendiente
                feats_ant = st.session_state.ultimo_features
                if feats_ant:
                    st.session_state.modelo.entrenar_o_actualizar(feats_ant, res)
                    st.session_state.resultado_pendiente = None

            # --- Predicción ---
            prob_subida = st.session_state.modelo.predecir(features)
            if prob_subida > 0.55:
                senal = 'COMPRA'
                fuerza = int(prob_subida * 100)
            elif prob_subida < 0.45:
                senal = 'VENTA'
                fuerza = int((1 - prob_subida) * 100)
            else:
                senal = None
                fuerza = 0

            st.session_state.senal_actual = senal
            st.session_state.fuerza_actual = fuerza
            st.session_state.confianza_actual = abs(prob_subida - 0.5) * 200

            # --- Señal en segundo 58 ---
            if senal and segundo_actual == 58:
                st.toast(f"🚀 SEÑAL: {senal} con fuerza {fuerza}%", icon="⏰")

            # --- Preparar entrenamiento futuro (al inicio de cada minuto) ---
            if segundo_actual == 0:
                if df_velas is not None and len(df_velas) >= 3:
                    ult_vela = df_velas.iloc[-2]
                    ant_vela = df_velas.iloc[-3]
                    if ult_vela['close'] > ant_vela['close']:
                        st.session_state.resultado_pendiente = 1
                    else:
                        st.session_state.resultado_pendiente = 0

    # ============================================
    # DISEÑO DE 3 COLUMNAS
    # ============================================
    col_izq, col_centro, col_der = st.columns([2, 2, 1.5])

    # --- COLUMNA IZQUIERDA: GRÁFICO DE VELAS ---
    with col_izq:
        st.markdown("### 📈 Gráfico en tiempo real")
        if st.session_state.activo_seleccionado and df_velas is not None and len(df_velas) >= 10:
            # Crear gráfico de velas con las últimas 10 velas
            df_graf = df_velas.iloc[-10:].copy()
            fig = go.Figure(data=[go.Candlestick(
                x=df_graf.index,
                open=df_graf['open'],
                high=df_graf['high'],
                low=df_graf['low'],
                close=df_graf['close'],
                increasing_line_color='#00FF88',
                decreasing_line_color='#FF4646'
            )])
            fig.update_layout(
                height=400,
                margin=dict(l=0, r=0, t=20, b=20),
                paper_bgcolor="#0A0C10",
                plot_bgcolor="#0A0C10",
                font_color="#E0E0E0",
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=False)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Cargando datos para el gráfico...")

    # --- COLUMNA CENTRAL: CÍRCULO DE SEÑAL ---
    with col_centro:
        st.markdown("### 💪 Señal actual")
        if st.session_state.senal_actual == 'COMPRA':
            color_circulo = "#00FF88"
            texto_circulo = "COMPRA"
        elif st.session_state.senal_actual == 'VENTA':
            color_circulo = "#FF4646"
            texto_circulo = "VENTA"
        else:
            color_circulo = "#2E2E2E"
            texto_circulo = "NEUTRO"

        # Círculo grande
        st.markdown(f"""
        <div style="background-color: {color_circulo}; width: 250px; height: 250px; border-radius: 50%; margin: 0 auto; display: flex; flex-direction: column; justify-content: center; align-items: center; border: 4px solid white; box-shadow: 0 0 20px {color_circulo};">
            <span style="color: white; font-size: 36px; font-weight: bold;">{texto_circulo}</span>
            <span style="color: white; font-size: 24px;">FUERZA: {st.session_state.fuerza_actual}%</span>
            <span style="color: white; font-size: 18px;">ABRE EN SEGUNDO 0</span>
        </div>
        """, unsafe_allow_html=True)

        # Información adicional debajo del círculo
        st.markdown(f"<p style='text-align: center;'>Confianza IA: {st.session_state.confianza_actual:.0f}%</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center;'>⏰ Hora actual: {ahora.strftime('%H:%M:%S')}</p>", unsafe_allow_html=True)
        if segundo_actual >= 58:
            st.warning("🚀 VENTANA DE SEÑAL ABIERTA (Segundos 58-59)")

    # --- COLUMNA DERECHA: PANEL DE ACTIVOS ---
    with col_der:
        st.markdown("### 📋 Activo en monitoreo")
        if st.session_state.activo_seleccionado:
            activo_nombre = st.session_state.activo_seleccionado.replace("-OTC", "")
            st.metric("Activo", activo_nombre)
            if st.session_state.ultimo_features:
                feats = st.session_state.ultimo_features
                st.metric("Precio", f"{feats['precio_actual']:.5f}")
                st.metric("Volumen (ratio)", f"{feats['volumen_ratio']:.2f}x")
                st.metric("Fuerza compra", f"{feats['fuerza_compra']:.1f}%")
                st.metric("RSI", f"{feats['rsi_14']:.1f}")
                st.metric("Momentum", f"{feats['precio_momentum']:.2f}%")
                st.metric("Volatilidad", f"{feats['log_volatilidad']:.3f}")
        else:
            st.info("Selecciona un activo en la barra lateral.")

        st.markdown("---")
        st.markdown("#### 📊 Otros activos OTC")
        # Mostrar una lista de los primeros 5 activos de la lista (simulada)
        if st.session_state.lista_activos:
            for act in st.session_state.lista_activos[:5]:
                st.write(f"- {act.replace('-OTC','')}")
        else:
            st.write("No hay activos disponibles.")

if __name__ == "__main__":
    main()
