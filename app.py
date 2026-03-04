"""
💪 INDICADOR FUERZA-IQ
Versión profesional con diseño de 3 columnas (gráfico, señal, activos).
Mercado OTC, vencimiento 1 minuto, señal en segundo 58.
Incluye IA RandomForest con entrenamiento continuo.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pytz
from datetime import datetime, timedelta
import time
import logging
import warnings
warnings.filterwarnings('ignore')

from streamlit_autorefresh import st_autorefresh
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# --- Configuración de logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Importar API de IQ Option ---
try:
    from iqoptionapi.stable_api import IQ_Option
    IQ_AVAILABLE = True
except ImportError:
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

# Autorefresh cada 1 segundo para precisión
st_autorefresh(interval=1000, key="segundo_a_segundo")

# --- Zona horaria (Ecuador) ---
ecuador_tz = pytz.timezone('America/Guayaquil')

# ============================================
# 1. CLASE DE CONEXIÓN MEJORADA
# ============================================
class IQOptionConnector:
    def __init__(self):
        self.api = None
        self.conectado = False
        self.balance = 0
        self.tipo_cuenta = "PRACTICE"
        self.activos_cache = []
        self.ultima_actualizacion_activos = 0
        self.email = None
        self.password = None

    def conectar(self, email, password):
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
            logging.info(f"Lista de activos OTC actualizada: {len(self.activos_cache)} encontrados.")
            return self.activos_cache
        except Exception as e:
            logging.error(f"Error obteniendo activos: {e}")
            return self.activos_cache if self.activos_cache else []

    def obtener_datos_vela_1min(self, activo, reintentos=2):
        if not self.conectado:
            return None
        for intento in range(reintentos):
            try:
                velas = self.api.get_candles(activo, 60, 15, time.time())
                if not velas or len(velas) < 3:
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
                if intento == reintentos-1:
                    return None
                time.sleep(2)
        return None

    def calcular_fuerza_y_volumen(self, df):
        if df is None or len(df) < 3:
            return None
        ult = df.iloc[-1]
        # Volumen ratio (vs media de 10 velas)
        vol_ma = df['volume'].iloc[-10:].mean()
        volumen_ratio = ult['volume'] / vol_ma if vol_ma > 0 else 1.0
        # Fuerza de compra aproximada
        rango = ult['high'] - ult['low']
        if rango == 0:
            fuerza_compra = 50
        else:
            fuerza_compra = 50 + 50 * ((ult['close'] - ult['open']) / rango)
        fuerza_compra = max(0, min(100, fuerza_compra))
        # Momentum
        if len(df) >= 3:
            precio_hace_3 = df['close'].iloc[-3]
            momentum = (ult['close'] - precio_hace_3) / precio_hace_3 * 100
        else:
            momentum = 0
        # RSI simple sobre 7 períodos
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=7, min_periods=1).mean().iloc[-1]
        avg_loss = loss.rolling(window=7, min_periods=1).mean().iloc[-1]
        rs = avg_gain / avg_loss if avg_loss != 0 else 100
        rsi = 100 - (100 / (1 + rs))
        # Log volatilidad
        retornos = df['close'].pct_change().dropna()
        log_vol = np.log1p(retornos.std()) if len(retornos) > 1 else 0
        # Posición en Bandas de Bollinger (simplificado)
        bb_std = df['close'].rolling(5).std().iloc[-1] * 2
        bb_media = df['close'].rolling(5).mean().iloc[-1]
        bb_sup = bb_media + bb_std
        bb_inf = bb_media - bb_std
        if bb_sup - bb_inf > 0:
            bb_pos = (ult['close'] - bb_inf) / (bb_sup - bb_inf)
        else:
            bb_pos = 0.5
        # Diferencia compra-venta (simulada)
        dif_compra_venta = 1.0 if ult['close'] > ult['open'] else -1.0

        return {
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

# ============================================
# 2. MODELO DE IA
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
                logging.info("Modelo actualizado.")
            self.X_buffer = []
            self.y_buffer = []

# ============================================
# 3. FUNCIONES AUXILIARES DE INTERFAZ
# ============================================
def crear_grafico_velas(df, activo):
    """Genera un gráfico de velas interactivo con Plotly."""
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        increasing_line_color='#00FF88',
        decreasing_line_color='#FF4646'
    )])
    fig.update_layout(
        title=f"{activo.replace('-OTC','')} - Últimas 10 velas",
        height=400,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="#0F1217",
        plot_bgcolor="#0F1217",
        font_color="#E0E0E0",
        xaxis=dict(showgrid=False, color="#AAAAAA"),
        yaxis=dict(showgrid=False, color="#AAAAAA"),
        hovermode="x unified"
    )
    return fig

# ============================================
# 4. INTERFAZ PRINCIPAL
# ============================================
def main():
    st.markdown("<h1 style='text-align: center; color:#00FF88;'>💪 INDICADOR FUERZA-IQ</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color:#AAAAAA;'>MERCADO OTC | SEÑAL EN SEGUNDO 58</h4>", unsafe_allow_html=True)
    st.markdown("---")

    # --- Inicializar estado de sesión ---
    if 'connector' not in st.session_state:
        st.session_state.connector = IQOptionConnector()
    if 'modelo' not in st.session_state:
        st.session_state.modelo = ModeloFuerza()
    if 'conectado' not in st.session_state:
        st.session_state.conectado = False
    if 'activo_actual' not in st.session_state:
        st.session_state.activo_actual = None
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
    if 'escaneando' not in st.session_state:
        st.session_state.escaneando = False
    if 'indice_activo' not in st.session_state:
        st.session_state.indice_activo = 0
    if 'log_estado' not in st.session_state:
        st.session_state.log_estado = ""

    # --- Barra lateral de configuración ---
    with st.sidebar:
        st.markdown("### 🔐 Acceso")
        if not st.session_state.conectado:
            email = st.text_input("Email", placeholder="tu@email.com")
            password = st.text_input("Contraseña", type="password", placeholder="••••••••")
            if st.button("🔌 Conectar", use_container_width=True):
                with st.spinner("Conectando..."):
                    ok, msg = st.session_state.connector.conectar(email, password)
                    if ok:
                        st.session_state.conectado = True
                        st.session_state.lista_activos = st.session_state.connector.obtener_activos_otc()
                        st.success("✅ Conectado")
                        st.rerun()
                    else:
                        st.error(f"❌ {msg}")
        else:
            st.success("✅ Conectado")
            saldo = st.session_state.connector.obtener_saldo()
            st.metric("Saldo", f"${saldo:.2f}")
            if st.button("🚪 Desconectar"):
                st.session_state.conectado = False
                st.session_state.escaneando = False
                st.rerun()

        if st.session_state.conectado:
            st.markdown("---")
            st.markdown("### ⚙️ Control")
            if not st.session_state.escaneando:
                if st.button("▶️ INICIAR ESCANEO", use_container_width=True):
                    st.session_state.escaneando = True
                    st.session_state.indice_activo = 0
                    st.session_state.log_estado = "🔍 Iniciando búsqueda de activos..."
                    st.rerun()
            else:
                if st.button("⏹️ DETENER ESCANEO", use_container_width=True):
                    st.session_state.escaneando = False
                    st.session_state.log_estado = "⏸️ Escaneo detenido."
                    st.rerun()

    # --- Si no está conectado, mostrar mensaje ---
    if not st.session_state.conectado:
        st.info("👆 Conéctate a IQ Option en la barra lateral para comenzar.")
        # Mostrar una vista previa
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.markdown("""
            <div style="background:#1E242C; border-radius:20px; padding:30px; text-align:center;">
                <i class="fas fa-plug" style="font-size:50px; color:#00FF88;"></i>
                <h3 style="color:#00FF88;">Esperando conexión...</h3>
            </div>
            """, unsafe_allow_html=True)
        return

    # --- Lógica de escaneo y análisis ---
    ahora = datetime.now(ecuador_tz)
    segundo_actual = ahora.second

    if st.session_state.escaneando:
        if not st.session_state.lista_activos:
            st.session_state.lista_activos = st.session_state.connector.obtener_activos_otc()
            if not st.session_state.lista_activos:
                st.warning("No se encontraron activos OTC disponibles.")
                st.session_state.escaneando = False

        if st.session_state.lista_activos:
            # Si no hay activo actual o terminamos de analizar, pasamos al siguiente
            if st.session_state.activo_actual is None:
                if st.session_state.indice_activo >= len(st.session_state.lista_activos):
                    st.session_state.indice_activo = 0  # reiniciamos ciclo
                activo = st.session_state.lista_activos[st.session_state.indice_activo]
                st.session_state.activo_actual = activo
                st.session_state.indice_activo += 1
                st.session_state.log_estado = f"🔍 Analizando {activo.replace('-OTC','')}..."

            # Obtener datos del activo actual
            df_velas = st.session_state.connector.obtener_datos_vela_1min(st.session_state.activo_actual)
            features = st.session_state.connector.calcular_fuerza_y_volumen(df_velas) if df_velas is not None else None

            if features is not None:
                st.session_state.ultimo_features = features

                # Entrenamiento con resultado de vela anterior
                if st.session_state.resultado_pendiente is not None:
                    st.session_state.modelo.entrenar_o_actualizar(
                        st.session_state.ultimo_features,
                        st.session_state.resultado_pendiente
                    )
                    st.session_state.resultado_pendiente = None

                # Predicción
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

                # Señal en segundo 58
                if senal and segundo_actual == 58:
                    st.toast(f"🚀 SEÑAL: {senal} con fuerza {fuerza}%", icon="⏰")

                # Preparar resultado para el próximo minuto
                if segundo_actual == 0 and df_velas is not None and len(df_velas) >= 3:
                    ult_vela = df_velas.iloc[-2]
                    ant_vela = df_velas.iloc[-3]
                    st.session_state.resultado_pendiente = 1 if ult_vela['close'] > ant_vela['close'] else 0

                # Si la fuerza es baja o no hay señal, pasamos al siguiente activo (cada cierto tiempo)
                # Aquí podemos decidir cambiar después de unos segundos sin señal fuerte
                if fuerza < 30 or (senal is None and segundo_actual % 10 == 0):
                    st.session_state.log_estado = f"⏭️ Fuerza insuficiente en {st.session_state.activo_actual.replace('-OTC','')} (fuerza {fuerza}%). Buscando otro..."
                    st.session_state.activo_actual = None
                    # Pequeña pausa para no saturar
                    time.sleep(1)
                    st.rerun()
            else:
                # No se pudieron obtener datos, pasamos al siguiente
                st.session_state.log_estado = f"⚠️ Error al obtener datos de {st.session_state.activo_actual.replace('-OTC','')}. Pasando al siguiente..."
                st.session_state.activo_actual = None
                time.sleep(1)
                st.rerun()
    else:
        # Si no está escaneando, no mostramos activo
        st.session_state.activo_actual = None

    # ============================================
    # DISEÑO DE 3 COLUMNAS
    # ============================================
    col_izq, col_centro, col_der = st.columns([2.2, 2, 1.8])

    # --- COLUMNA IZQUIERDA: GRÁFICO DE VELAS ---
    with col_izq:
        st.markdown("### 📈 Gráfico en tiempo real")
        if st.session_state.activo_actual and st.session_state.ultimo_features:
            df_graf = st.session_state.connector.obtener_datos_vela_1min(st.session_state.activo_actual)
            if df_graf is not None and len(df_graf) >= 5:
                fig = crear_grafico_velas(df_graf.iloc[-10:], st.session_state.activo_actual)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Cargando datos para el gráfico...")
        else:
            st.info("Activa el escaneo para ver el gráfico.")

    # --- COLUMNA CENTRAL: CÍRCULO DE SEÑAL ---
    with col_centro:
        st.markdown("### 💪 Señal actual")
        if st.session_state.senal_actual == 'COMPRA':
            color = "#00FF88"
            texto = "COMPRA"
        elif st.session_state.senal_actual == 'VENTA':
            color = "#FF4646"
            texto = "VENTA"
        else:
            color = "#2E2E2E"
            texto = "NEUTRO"

        # Círculo
        st.markdown(f"""
        <div style="background-color: {color}; width: 240px; height: 240px; border-radius: 50%; margin: 0 auto; display: flex; flex-direction: column; justify-content: center; align-items: center; border: 4px solid white; box-shadow: 0 0 25px {color};">
            <span style="color: white; font-size: 36px; font-weight: bold;">{texto}</span>
            <span style="color: white; font-size: 26px;">{st.session_state.fuerza_actual}%</span>
            <span style="color: white; font-size: 16px;">ABRE EN SEGUNDO 0</span>
        </div>
        """, unsafe_allow_html=True)

        # Información adicional
        st.markdown(f"<p style='text-align: center;'>Confianza IA: {st.session_state.confianza_actual:.0f}%</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center;'>⏰ Hora: {ahora.strftime('%H:%M:%S')}</p>", unsafe_allow_html=True)
        if segundo_actual >= 58:
            st.warning("🚀 VENTANA DE SEÑAL ABIERTA (seg 58-59)")

        # Mostrar el estado de escaneo
        if st.session_state.log_estado:
            st.info(st.session_state.log_estado)

    # --- COLUMNA DERECHA: PANEL DE ACTIVOS ---
    with col_der:
        st.markdown("### 📋 Activo en monitoreo")
        if st.session_state.activo_actual:
            nombre = st.session_state.activo_actual.replace("-OTC", "")
            st.metric("Activo", nombre)
            if st.session_state.ultimo_features:
                feats = st.session_state.ultimo_features
                st.metric("Precio", f"{feats['precio_actual']:.5f}")
                st.metric("Volumen (ratio)", f"{feats['volumen_ratio']:.2f}x")
                st.metric("Fuerza compra", f"{feats['fuerza_compra']:.1f}%")
                st.metric("RSI", f"{feats['rsi_14']:.1f}")
        else:
            st.info("Ningún activo en análisis.")

        st.markdown("---")
        st.markdown("#### 📊 Activos OTC disponibles")
        if st.session_state.lista_activos:
            for act in st.session_state.lista_activos[:8]:
                st.write(f"- {act.replace('-OTC','')}")
        else:
            st.write("Cargando lista...")

if __name__ == "__main__":
    main()
