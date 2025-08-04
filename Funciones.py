import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import geopandas as gpd 
import seaborn as sns 
import networkx as nx 
import scipy as scipy


############################ PARTE (nueva) TP 2 ######################################################################################################################

def calcula_L(A):# La función recibe la matriz de adyacencia A y calcula la matriz laplaciana
    grados = np.diag(A.sum(axis=1))
    return grados - A

def calcula_R(A):# La funcion recibe la matriz de adyacencia A y calcula la matriz de modularidad
    grados = A.sum(axis=1)
    m = grados.sum() / 2
    K = np.outer(grados, grados)
    return A - K / (2 * m)

def calcula_lambda(L, v): # Recibe L y v y retorna el corte asociado
    return float(v.T @ L @ v) / 4   #> Les falto dividir por 4

def calcula_Q(R, v):# La funcion recibe R y s y retorna la modularidad (a menos de un factor 2E)
    return float(v.T @ R @ v)

#> Es bastante mala practica hacer esto de esta forma y encima les da mal porque metpot1 usa la propiedad .shape y 
#> ustedes no pusieron esa propiedad en esta clase. Calculen la inversa y listo! 

def metpot1(A, tol=1e-8, maxrep=np.inf): # Recibe una matriz A y calcula su autovalor de mayor módulo, con un error relativo menor a tol y-o haciendo como mucho maxrep repeticiones
    n = A.shape[0]
    v = np.random.uniform(-1, 1, size=n) # Generamos un vector de partida aleatorio, entre -1 y 1
    v = v / np.linalg.norm(v) # Lo normalizamos
    v1 = A @ v # Aplicamos la matriz una vez
    v1 = v1 / np.linalg.norm(v1) # normalizamos
    l = np.dot(v, A @ v) # Calculamos el autovector estimado
    l1 = np.dot(v1, A @ v1) # Y el estimado en el siguiente paso
    nrep = 0
    while np.abs(l1 - l) / np.abs(l) > tol and nrep < maxrep: # Si estamos por debajo de la tolerancia buscada 
        v = v1 # actualizamos v y repetimos
        l = l1
        v1 = A @ v # Calculo nuevo v1
        v1 = v1 / np.linalg.norm(v1) # Normalizo
        l1 = np.dot(v1, A @ v1) # Calculo autovector
        nrep += 1
    if not nrep < maxrep:
        print("MaxRep alcanzado")
    l = np.dot(v, A @ v)  # Calculamos el autovalor
    return v1, l, nrep < maxrep  # autovector, autovalor, converge



def deflaciona(A, tol=1e-8, maxrep=np.inf):
    # Recibe la matriz A, una tolerancia para el metodo de la potencia, y un numero maximo de repeticiones
    v1, l1, _ = metpot1(A, tol, maxrep)  # Buscamos primer autovector con método de la potencia

    # Deflacion: restamos el componente dominante
    deflA = A - l1 * np.outer(v1, v1) / np.dot(v1, v1)

    return deflA

def metpot2(A, v1, l1, tol=1e-8, maxrep=np.inf):   # La funcion aplica el metodo de la potencia para buscar el segundo autovalor de A, suponiendo que sus autovectores son ortogonales
   # v1 y l1 son los primeors autovectores y autovalores de A}
    deflA = A - l1 * np.outer(v1, v1) / np.dot(v1, v1)
    return metpot1(deflA, tol, maxrep)


def metpotI(A, mu, tol=1e-8, maxrep=np.inf):
    A_mu = A + mu * np.eye(A.shape[0])
    L, U = calculaLU(A_mu)

    def matvec(v):
        y = scipy.linalg.solve_triangular(L, v, lower=True)
        return scipy.linalg.solve_triangular(U, y)

    return metpot1(matvec, A.shape[0], tol=tol, maxrep=maxrep)


def metpotI2(A, mu, tol=1e-8, maxrep=np.inf):
    # Recibe la matriz A, y un valor mu y retorna el segundo autovalor y autovector de la matriz A,
    # suponiendo que sus autovalores son positivos excepto por el menor que es igual a 0
    # Retorna el segundo autovector, su autovalor, y si el metodo llegó a converger.
    X = A + mu * np.eye(A.shape[0])   # Matriz A shifteada
    L, U = calculaLU(X)               # LU de la inversa de (A + mu·I)

    def matvec(v):
        y = scipy.linalg.solve_triangular(L, v, lower=True) # Resuelve L·y = v
        return scipy.linalg.solve_triangular(U, y)          # Luego U·x = y , obtenemos x = X⁻¹ @ v

    # Construimos matriz inversa simulada aplicando matvec por columnas de la identidad
    n = A.shape[0]
    I = np.eye(n)
    X_inv = np.zeros_like(A, dtype=float)
    for i in range(n):
        X_inv[:, i] = matvec(I[:, i])

    # Deflacionamos la inversa
    deflX = deflaciona(X_inv, tol=tol, maxrep=maxrep)

    # Método de la potencia sobre matriz deflacionada
    v, l, conv = metpot1(deflX, tol=tol, maxrep=maxrep)

    # Deshacemos inversion y shift
    l = 1 / l
    l -= mu

    return v, l, conv



def laplaciano_iterativo(A, niveles, nombres_s=None):
    # Recibe una matriz A, una cantidad de niveles sobre los que hacer cortes, y los nombres de los nodos
    # Retorna una lista con conjuntos de nodos representando las comunidades.
    # La función debe, recursivamente, ir realizando cortes y reduciendo en 1 el número de niveles hasta llegar a 0 y retornar.
    if nombres_s is None: # Si no se proveyeron nombres, los asignamos poniendo del 0 al N-1
        nombres_s = range(A.shape[0])
    if A.shape[0] == 1 or niveles == 0: # Si llegamos al último paso, retornamos los nombres en una lista
        return([nombres_s])
    else: # Sino:
        L = calcula_L(A) # Recalculamos el L
        mu = A.shape[0]
        v, l, _ = metpotI2(L, mu=mu) # Encontramos el segundo autovector de L
        # Recortamos A en dos partes, la que está asociada a el signo positivo de v y la que está asociada al negativo
        Ap = A[np.ix_([i for i, vi in enumerate(v) if vi > 0], [i for i, vi in enumerate(v) if vi > 0])] # Asociado al signo positivo
        Am = A[np.ix_([i for i, vi in enumerate(v) if vi < 0], [i for i, vi in enumerate(v) if vi < 0])] # Asociado al signo negativo

        return(
                laplaciano_iterativo(Ap, niveles - 1,
                                     nombres_s=[ni for ni, vi in zip(nombres_s, v) if vi > 0]) +
                laplaciano_iterativo(Am, niveles - 1,
                                     nombres_s=[ni for ni, vi in zip(nombres_s, v) if vi < 0])
              )



def modularidad_iterativo(A=None, R=None, nombres_s=None):
    # Recibe una matriz A, una matriz R de modularidad, y los nombres de los nodos
    # Retorna una lista con conjuntos de nodos representando las comunidades.

    if A is None and R is None:
        print('Dame una matriz')
        return(np.nan)
    if R is None:
        R = calcula_R(A)
    if nombres_s is None:
        nombres_s = range(R.shape[0])
    # Acá empieza lo bueno
    if R.shape[0] == 1: # Si llegamos al último nivel
        return([nombres_s])
    else:
        v, l, _ = metpot1(R) # Primer autovector y autovalor de R
        # Modularidad Actual:
        Q0 = np.sum(R[v > 0, :][:, v > 0]) + np.sum(R[v < 0, :][:, v < 0])
        if Q0 <= 0 or all(v > 0) or all(v < 0): # Si la modularidad actual es menor a cero, o no se propone una partición, terminamos
            return([nombres_s])
        else:
            ## Hacemos como con L, pero usando directamente R para poder mantener siempre la misma matriz de modularidad
            Rp = R[np.ix_([i for i, vi in enumerate(v) if vi > 0], [i for i, vi in enumerate(v) if vi > 0])] # Parte de R asociada a los valores positivos de v
            Rm = R[np.ix_([i for i, vi in enumerate(v) if vi < 0], [i for i, vi in enumerate(v) if vi < 0])] # Parte asociada a los valores negativos de v
            vp, lp, _ = metpot1(Rp)  # autovector principal de Rp
            vm, lm, _ = metpot1(Rm) # autovector principal de Rm

            # Calculamos el cambio en Q que se produciría al hacer esta partición
            Q1 = 0
            if not all(vp > 0) or all(vp < 0):
               Q1 = np.sum(Rp[vp > 0, :][:, vp > 0]) + np.sum(Rp[vp < 0, :][:, vp < 0])
            if not all(vm > 0) or all(vm < 0):
                Q1 += np.sum(Rm[vm > 0, :][:, vm > 0]) + np.sum(Rm[vm < 0, :][:, vm < 0])
            if Q0 >= Q1: # Si al partir obtuvimos un Q menor, devolvemos la última partición que hicimos
                return([[ni for ni, vi in zip(nombres_s, v) if vi > 0],
                        [ni for ni, vi in zip(nombres_s, v) if vi < 0]])
            else:
                # Sino, repetimos para los subniveles
                return(
                    modularidad_iterativo(R=Rp,
                                          nombres_s=[ni for ni, vi in zip(nombres_s, v) if vi > 0]) +
                    modularidad_iterativo(R=Rm,
                                          nombres_s=[ni for ni, vi in zip(nombres_s, v) if vi < 0])
                )

###########################################################  PARTE 1 (corregida) TP  #####################################################################



def construye_adyacencia(D,m):
    # Función que construye la matriz de adyacencia del grafo de museos
    # D matriz de distancias, m cantidad de links por nodo
    # Retorna la matriz de adyacencia como un numpy.
    D = D.copy()
    l = [] # Lista para guardar las filas
    for fila in D: # recorriendo las filas, anexamos vectores lógicos
        l.append(fila<=fila[np.argsort(fila)[m]] ) # En realidad, elegimos todos los nodos que estén a una distancia menor o igual a la del m-esimo más cercano
    A = np.asarray(l).astype(int) # Convertimos a entero
    np.fill_diagonal(A,0) # Borramos diagonal para eliminar autolinks
    return(A)

def calculaLU(matriz):
    A = matriz                        # matriz original
    n = A.shape[0]
    U = A.copy()                      # U parte como copia de A (la vamos modificando con cada iteracion)
    L = np.eye(n)                     # L parte como matriz identidad (tamaño n)

    for j in range(n):               # Recorremos columnas j
        for i in range(j + 1, n):    # Recorremos filas debajo del pivote (problema si es 0)
            L[i, j] = U[i, j] / U[j, j]                   # actualizamos L poniendo el que sera el "multiplicador" que nos dira por cuanto debemos multiplicar la fila anterior para que se anunle la siguiente
            U[i, :] = U[i, :] - L[i, j] * U[j, :]         # Eliminamos la fila usando lo anterior

    return L, U


def calcula_matriz_C(A):
    K_diag = A.sum(axis=1)                  # (suma por fila)
    K_inv = np.diag(1 / K_diag)             # Inversa de K (es diagonal osea 1/diag de k)
    C = A.T @ K_inv                         # C tiene columnas estocásticas (suman 1 ya q son probabilidades de ir a x museo desde h museo). Ya tenemos nuestra matriz de trancisiones a partir de A^T * K^-1
    return C

def calcula_pagerank(A, alpha):
    N = A.shape[0]
    C = calcula_matriz_C(A)
    I = np.eye(N)
    M = (N / alpha) * (I - (1 - alpha) * C) #hago el calculo de M. (Alpha es la contemplacion de que un usuario vaya a un museo random en vez de uno cercano)
    b = np.ones(N)

    # obtengo LU
    L, U = calculaLU(M)

    # Resolución del sistema: {Ly = b
    #                         {Up = y
    Up = scipy.linalg.solve_triangular(L, b, lower=True) #Primera inversión usando L
    p = scipy.linalg.solve_triangular(U, Up) # Segunda inversión usando U


    return p

def calcula_matriz_C_continua(D):
    # Función para calcular la matriz de trancisiones C
    # A: Matriz de adyacencia
    # Retorna la matriz C en versión continua
    D = D.copy()
    np.fill_diagonal(D,1) # llenamos de 1s para que no divida por Cero.
    F = 1 / D
    np.fill_diagonal(F, 0) # Luego llenamos la diagonal de F de ceros para que los museos no se autoreferencien.
    Kinv = np.diag(1 / F.sum(axis=1)) # Calcula inversa de la matriz K, que tiene en su diagonal la suma por filas de F

    ###### ACA LA CORRECCION (el resto bien)###################
    C = F @ Kinv # Calcula C multiplicando Kinv y F 
    ############################################################
    return C

def calcula_B(C, cantidad_de_visitas):
   #en este caso cantidad_de_visitas = r que hablamos anteriormente
    B = np.eye(C.shape[0])  # C^0 = I (osea arranco estando en mi museo de origen sin moverme)
    C_potencia = np.eye(C.shape[0])
    for _ in range(1, cantidad_de_visitas): #desde 1 hasta r-1
        C_potencia = C_potencia @ C #multiplico a c por si mismo hasta r-1
        B += C_potencia
    return B #cada elemento (i,j) de B nos dira la cantidad de veces que partiendo de j una persona termino en i habiendo hecho r "pasos"
