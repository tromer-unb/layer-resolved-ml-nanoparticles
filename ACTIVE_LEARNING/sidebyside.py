import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# ==================================================
# ARQUIVOS
# ==================================================
DESCR_FILE = "descriptors.csv"      # descritores (na mesma ordem das estruturas)
ENERGY_FILE = "ENERGIAS_norm.csv"   # Etot na mesma ordem
OUT_FILE = "file.csv"

# ==================================================
# 1. LEITURA
# ==================================================
df_desc = pd.read_csv(DESCR_FILE)
df_energy = pd.read_csv(ENERGY_FILE, skipinitialspace=True)

# ==================================================
# 2. LIMPA NOMES DE COLUNAS
# ==================================================
df_desc.columns = df_desc.columns.str.strip()
df_energy.columns = df_energy.columns.str.strip()

# ==================================================
# 3. CHECK CRÍTICO: ORDEM = ESTRUTURAS
# ==================================================
if len(df_desc) != len(df_energy):
    raise ValueError(
        f"❌ ERRO: tamanhos diferentes | descritores={len(df_desc)} energias={len(df_energy)}"
    )

# ==================================================
# 4. CONVERSÃO PARA NUMÉRICO
# ==================================================
df_desc = df_desc.apply(pd.to_numeric, errors="coerce")
df_energy["Etot"] = pd.to_numeric(df_energy["Etot"], errors="coerce")

# ==================================================
# 5. CONCATENA POR ORDEM (LINHA i = i.xyz)
# ==================================================
df = pd.concat(
    [
        df_desc.reset_index(drop=True),
        df_energy[["Etot"]].reset_index(drop=True)
    ],
    axis=1
)

# ==================================================
# 6. REMOVE COLUNAS NÃO NUMÉRICAS
# ==================================================
df = df.select_dtypes(include=[np.number])

# ==================================================
# 7. NaN → 0.0
# ==================================================
df = df.fillna(0.0)

# ==================================================
# 8. CRIA ID = NÚMERO DA ESTRUTURA (1.xyz, 2.xyz, ...)
# ==================================================
df.insert(0, "id", np.arange(1, len(df) + 1))

# ==================================================
# 9. Z-SCORE (NÃO NORMALIZA id NEM Etot)
# ==================================================
features = df.columns.difference(["id", "Etot"])

scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

# ==================================================
# 10. SALVA
# ==================================================
df.to_csv(OUT_FILE, index=False)

# ==================================================
# 11. CHECK FINAL
# ==================================================
print("✅ file.csv criado corretamente")
print("IDs:", df["id"].min(), "→", df["id"].max())
print("Shape:", df.shape)
print(df.head())
