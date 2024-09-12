import numpy as np
import matplotlib.pyplot as plt

titulo = "channels.vs.energy.txt"
data = np.loadtxt(titulo)
print(titulo)

X = data[:, 2] #ENERGIA
Y = data[:, 0] #CHANNELS
Y_err = data[:, 3] #error/incertidumbre en mu

# Ajuste de grado N=2 con covarianza
coef, cov = np.polyfit(X, Y, 2, cov=True)
polinomio = np.poly1d(coef)

# Valores ajustados. Curva de tendencia
x_ajustado = np.linspace(X.min(), X.max(), 100)
y_ajustado = polinomio(x_ajustado)

# Mostrar los resultados
print(f"A: {coef[0]} ± {np.sqrt(cov[0, 0])}")
print(f"B: {coef[1]} ± {np.sqrt(cov[1, 1])}")
print(f"C: {coef[2]} ± {np.sqrt(cov[2, 2])}")

# Valores ajustados/predecidos
Y_pred = polinomio(X)

# Calcular chi cuadrado reducido
red_chi_squared = np.sum(((Y - Y_pred)/Y_err) ** 2)
print(f"Chi cuadrado reducido (χr²): {red_chi_squared}")


"""
plt.plot(x_ajustado, y_ajustado, color='black', linewidth=1, label='Fit')
#plt.scatter(X, Y, color='blue', label="Background lines", s=15)
plt.scatter(X[25:42], Y[25:42], color='teal', label="Background lines", s=15)
plt.scatter(X[0:2], Y[0:2], color='blue', label='60Co', s=15)
plt.scatter(X[2], Y[2], color='red', label='137Cs', s=15)
plt.scatter(X[6:20], Y[6:20], color='deeppink', label='152Eu', s=15)
plt.scatter(X[20:25], Y[20:25], color='orange', label='133Ba', s=15)
plt.scatter(X[3:6], Y[3:6], color='limegreen', label='207Bi', s=15)
plt.xlabel('Channel')
plt.ylabel('Energy (keV)')
plt.title('Energy calibration')
equation = f" E = {4.026e-6}·Ch² + {round(coef[1],4)}·Ch + {round(coef[2],4)}"
plt.text(10, 2100, equation, fontsize=12, color='black')
plt.legend()
plt.show()
"""
# Gráfico de residuos
residuos = Y - Y_pred
plt.figure()
#plt.scatter(X, residuos, color='blue')
plt.scatter(X[25:42], residuos[25:42], color='teal', label="Background lines", s=15)
plt.scatter(X[0:2], residuos[0:2], color='blue', label='60Co', s=15)
plt.scatter(X[2], residuos[2], color='red', label='137Cs', s=15)
plt.scatter(X[6:20], residuos[6:20], color='deeppink', label='152Eu', s=15)
plt.scatter(X[20:25], residuos[20:25], color='orange', label='133Ba', s=15)
plt.scatter(X[3:6], residuos[3:6], color='limegreen', label='207Bi', s=15)
plt.hlines(0, min(X), max(X), colors='black', linestyles='dashed')
plt.xlabel('Channels')
plt.ylabel('Residuals')
plt.title("Residual Analysis for energy calibration")
plt.legend()
plt.grid(True)
plt.show()
