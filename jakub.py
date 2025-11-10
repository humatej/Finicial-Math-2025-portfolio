import yfinance as yf # používa sa na sťahovanie historických dát z Yahoo Finance
import pandas as pd # poskytuje formát pre prácu s tabuľkovými dátami
import matplotlib.pyplot as plt # knižnica na kreslenie grafov a časových radov
import numpy as np # slúži na numerické výpočty a prácu s poľami
import cvxopt as cvx # používa sa na riešenie úloh kvadratického programovania

"""Markowitz s rieseniami.ipynb

# Markowitzova teória portfólia

## 1 Stiahnutie dát

Najprv načítajte potrebné balíky. Ak ich nemáte, nainštalujte si ich.
"""
"""Prikaz `yf.download` stiahne historické dáta o konkrétnej akcii alebo viacerých akciách. Obsahuje nasledujúce parametre:
- tickers - skratka akcie (napr. "TSLA"), môže byť aj zoznam viacerých tickerov,
- start - dátum začiatku ("YYYY-MM-DD"),
- end - dátum konca ("YYYY-MM-DD"),
- interval - interval dát ("1d" = denne, "1wk" = týždenne, "1mo" = mesačne; voliteľné),
- group_by - spôsob zoskupenia stĺpcov pri viacerých tickeroch ("ticker" alebo "column"; voliteľné),
- auto_adjust - True/False, či sa Close upraví o dividendy a splity (voliteľné), použitím upravenej ceny získame presnejší obraz o reálnej výkonnosti akcie v čase.

#### **Príklad**: Stiahnite dáta pre cenu akcie Tesla Inc. z obdobia od 1.1.2023 do 31.12.2024.
"""

tsla = yf.download("TSLA", start="2023-01-01", end="2024-12-31", auto_adjust=True)

"""Dáta sú uložené v premennej `tsla` vo formáte `DataFrame` z knižnice `pandas`, ktorý umožňuje jednoduchú manipuláciu s dátami. Funkcia `head()` zobrazí prvých päť riadkov:"""

print(tsla.head())

"""Ak nás zaujíma iba záverečná cena (Close), môžeme vypísať len tento stĺpec:

"""

print(tsla['Close'].head())

"""Pre lepšiu predstavu o vývoji ceny akcie môžeme záverečnú cenu (Close) graficky znázorniť:


"""

plt.figure(figsize=(10,6))
plt.plot(tsla['Close'], color='blue')
plt.title("Vývoj ceny akcie TSLA")
plt.xlabel("Dátum")
plt.ylabel("Cena")
plt.show()

"""#### **Príklad**: Stiahnite dáta pre ceny nasledujúcich piatich akcií:
- Adidas AG (ADDYY),
- Apple Inc. (AAPL),
- Meta Platforms, Inc. (META),
- Starbucks Corporation (SBUX),
- Microsoft Corporation (MSFT)

z obdobia od 1.10.2023 do 30.9.2025.
"""

tickers = ["ADDYY", "AAPL", "META", "SBUX", "MSFT"]

data = yf.download(tickers, start="2023-10-01", end="2025-09-30", auto_adjust=True)['Close']

data.head()

"""## 2 Spracovanie dát

Odstránenie riadkov (dátumov) s chýbajúcimi hodnotami:
"""

data = data.dropna()

"""Vykreslenie vývoja cien jednotlivých akcií:"""

plt.figure(figsize=(12,6))

plt.plot(data.index, data['ADDYY'], color='black', label='ADDYY')
plt.plot(data.index, data['AAPL'], color='green', label='AAPL')
plt.plot(data.index, data['META'], color='orange', label='META')
plt.plot(data.index, data['SBUX'], color='purple', label='SBUX')
plt.plot(data.index, data['MSFT'], color='red', label='MSFT')

plt.xlabel("Dátum")
plt.ylabel("Cena")
plt.title("Vývoj cien akcií")
plt.legend(loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

"""Je praktickejšie pracovať s výnosmi akcií ako s ich cenami. Preto vypočítame z cien akcií ich výnosy:

\begin{equation}
r_t = \frac{X_t - X_{t-1}}{X_{t-1}} = \frac{X_t}{X_{t-1}} - 1.
\end{equation}

V Pythone môžeme výnosy vypočítať ako
"""

r = data/data.shift(1) - 1  # data.shift(1) posunie hodnoty o jeden riadok nadol
r = r.dropna()  # odstráni prvý riadok s NaN
r.head()

"""Vykreslenie vývoja výnosov jednotlivých akcií:"""

plt.figure(figsize=(12,6))

plt.plot(data.index[1::], r['ADDYY'], color='black', label='ADDYY')
plt.plot(data.index[1::], r['AAPL'], color='green', label='AAPL')
plt.plot(data.index[1::], r['META'], color='orange', label='META')
plt.plot(data.index[1::], r['SBUX'], color='purple', label='SBUX')
plt.plot(data.index[1::], r['MSFT'], color='red', label='MSFT')

plt.xlabel("Dátum")
plt.ylabel("Výnos")
plt.title("Vývoj výnosov akcií")
plt.legend(loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

"""### 2.1 Očakávané výnosy

Denný očakávaný výnos akcie vypočítame ako priemer jej denných výnosov:


\begin{equation}
\bar{r} = \frac{1}{N} \sum_{t=1}^{N} r_t.
\end{equation}

Pre päť akcií dostaneme vektor:
"""

r_mean = r.mean() # odhad denných očakávaných výnosov
print(r_mean)

"""Ročné očakávané výnosy získame vynásobením denných očakávaných výnosov hodnotou 252 (počet obchodovacích dní v roku):"""

r_mean = r_mean * 252
print(r_mean)

"""### 2.2 Volatilita výnosov

Výberová kovariancia medzi výnosmi dvoch akcií $x$ a $y$ sa počíta ako  

\begin{equation}
\text{cov}(x, y) = \frac{1}{T-1} \sum_{t=1}^{T} (x_t - \bar{x})(y_t - \bar{y}).
\end{equation}

Pre päť akcií dostaneme maticu:
"""

V = np.cov(r, rowvar = False) # rowvar = False znamená, že stĺpce berieme ako premenné
print(V)

"""Kovariančnú maticu ročných výnosov získame škálovaním dennej kovariancie:"""

V = V * 252
print(V)

"""Volatility výnosov jednotlivých akcií vypočítame pomocou funkcií `diag()` a `sqrt()`:"""

sigma = np.sqrt(np.diag(V))
print(sigma)

"""### 2.3 Naivne diverzifikované portfólio

Najprv odhadneme ročný očakávaný výnos a volatilitu ročného výnosu naivne diverzifikovaného portfólia.

Váhy v naivne diverzifikovanom portfóliu sú rovnaké pre všetky akcie. Pri vektorovom alebo maticovom násobení sa v Pythone používa operátor @ alebo funkcia `np.dot()`.
"""

n = data.shape[1] # počet akcií
w_naive = np.repeat(1/n, n) # rovnaké váhy

r_naive = w_naive @ r_mean

sigma_naive = np.sqrt(w_naive @ V @ w_naive)

print(r_naive)
print(sigma_naive)

"""## 3 Markowitzov model

Markowitzov problém budeme najprv riešiť analyticky a potom numericky pomocou funkcie `cvx.solvers.qp`.

#### **Príklad 1**: Nájdite analytické riešenie Markowitzovho problému, pričom použite vektor očakávaných ročných výnosov $\bar{r}$, kovariančnú maticu ročných výnosov $V$ a vhodne zvoľte požadovanú hodnotu očakávaného ročného výnosu portfólia $\bar{r}_p$.
"""

n = data.shape[1]
r_p = 0.10

A = np.ones(n) @ np.linalg.inv(V) @ r_mean
B = r_mean @ np.linalg.inv(V) @ r_mean
C = np.ones(n) @ np.linalg.inv(V) @ np.ones(n)
D = B*C - A**2

g = (B*(np.linalg.inv(V)@np.ones(n)) - A*(np.linalg.inv(V)@r_mean))/D
h = (C*(np.linalg.inv(V)@r_mean) - A*(np.linalg.inv(V)@np.ones(n)))/D

w_analytic = g + h*r_p
sigma_analytic = np.sqrt(w_analytic.T @ V @ w_analytic)

print(w_analytic)
print(sigma_analytic)

"""#### **Príklad 2**: Nájdite numerické riešenie Markowitzovho problému pomocou funkcie `solvers.qp()` z balíka `cvxopt`. Porovnajte získané numerické riešenie s analytickým riešením z predchádzajúceho príkladu.

Funkcia `solvers.qp(P,q,G,h,A,b)` rieši optimalizačnú úlohu s kvadratickou účelovou funkciou v tvare

$$ \min_{x} \, \frac{1}{2}x^{T}Px + q^{T}x$$

s ohraničeniami $$ Gx \leq h, $$
$$ Ax = b. $$
"""

n = data.shape[1]
r_p = 0.1

# ucelova funkcia
P = cvx.matrix(V) # kovariancna matica (kvadraticky clen)
q = cvx.matrix(np.zeros(n)) # nulovy linearny clen

# ohranicenia
G = cvx.matrix(0.0, (n,n)) # matica z ohranicenia typu nerovnost
h = cvx.matrix(0.0, (n,1))

A = cvx.matrix(np.array([r_mean, np.ones(n)])) # matica z ohraniceni typu rovnost
b = cvx.matrix([r_p, 1])

# riesenie problemu kvadratickeho programovania
sol = cvx.solvers.qp(P, q, G, h, A, b)

weights_numerical = np.array(sol['x']).flatten()
volatility_numerical_1 = np.sqrt(weights_numerical @ V @ weights_numerical)

print(weights_numerical)
print(volatility_numerical_1)

"""#### **Príklad 3**: Uvažujte ohraničenie na nezápornosť váh, t. j. $w_i \geq 0$  pre všetky $i$. Nájdite numerické riešenie Markowitzovho problému pomocou funkcie `solvers.qp()`."""

n = data.shape[1]
r_p = 0.1

# ucelova funkcia
P = cvx.matrix(V) # kovariancna matica (kvadraticky clen)
q = cvx.matrix(np.zeros(n)) # nulovy linearny clen

# ohranicenia
G = cvx.matrix(-np.eye(n)) # matica z ohraniceni typu nerovnost
h = cvx.matrix(np.zeros(n))

A = cvx.matrix(np.array([r_mean, np.ones(n)])) # matica z ohraniceni typu rovnost
b = cvx.matrix([r_p, 1])

# riesenie problemu kvadratickeho programovania
sol = cvx.solvers.qp(P, q, G, h, A, b)

weights_numerical = np.array(sol['x']).flatten()
volatility_numerical_2 = np.sqrt(weights_numerical @ V @ weights_numerical)

print(weights_numerical)
print(volatility_numerical_2)

"""#### **Príklad 4**: Do roviny $\sigma_p$-$\bar{r}_p$ vykreslite hranicu s povolenými krátkymi pozíciami pre očakávané výnosy portfólia $\bar{r}_p$ vo vhodne zvolenom rozsahu. Do toho istého grafu zakreslite hranicu so zakázanými krátkymi pozíciami pre celý možný rozsah $\bar{r}_p$.

Pomocou funkcie `np.arange(start,stop,step)` môžeme rozdeliť ľubovoľný interval na podintervaly rovnakej dĺžky.
"""

n = data.shape[1]

# rozsah požadovaných výnosov portfólia
r_p_short = np.arange(0, 0.8, 0.0001) # pre kratke pozicie
r_p_long = np.arange(np.min(r_mean) + 0.0001, np.max(r_mean) - 0.0001, 0.0001) # pre dlhe pozicie

sigma_short = np.zeros(len(r_p_short))
sigma_long  = np.zeros(len(r_p_long))

# hranica s kratkymi poziciami
for i in range(len(r_p_short)):
    P = cvx.matrix(V)
    q = cvx.matrix(np.zeros(n))

    G = cvx.matrix(0.0, (n,n))
    h = cvx.matrix(0.0, (n,1))

    A = cvx.matrix(np.array([r_mean, np.ones(n)]))
    b = cvx.matrix([r_p_short[i], 1])

    sol = cvx.solvers.qp(P, q, G, h, A, b)
    sigma_short[i] = np.sqrt(sol['primal objective']*2)


# hranica s dlhymi poziciami
for i in range(len(r_p_long)):
    P = cvx.matrix(V)
    q = cvx.matrix(np.zeros(5))

    G = cvx.matrix(-np.eye(5))
    h = cvx.matrix(np.zeros(5))

    A = cvx.matrix(np.vstack([r_mean, np.ones(n)]))
    b = cvx.matrix([r_p_long[i], 1])

    sol = cvx.solvers.qp(P, q, G, h, A, b)
    sigma_long[i] = np.sqrt(sol['primal objective']*2)


# vykreslenie
plt.figure(figsize=(10,6))
plt.plot(sigma_short, r_p_short, label="hranica s kratkymi poziciami", linewidth=2)
plt.plot(sigma_long, r_p_long, label="hranica s dlhymi poziciami", linewidth=2)

plt.xlabel(r"$\sigma_p$ (volatilita portfolia)")
plt.ylabel(r"$\bar{r}_p$ (očakávaný výnos portfolia)")
plt.grid(True)
plt.legend()
plt.show()

"""#### **Príklad 5**: Do roviny $\sigma_p$-$\bar{r}_p$ pridajte nájdené optimálne portfólia z predchádzajúcich príkladov 2 a 3. Taktiež vykreslite uvažovaných päť akcií a naivne diverzifikované porfólio."""

import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(10,6))

# hranica s kratkymi poziciami
plt.plot(sigma_short, r_p_short, label="hranica s kratkymi poziciami", color="black")
plt.scatter(volatility_numerical_1, r_p, marker="o", color="red")

# hranica s dlhymi poziciami
plt.plot(sigma_long, r_p_long, label="hranica s dlhymi poziciami", color="green")
plt.scatter(volatility_numerical_2, r_p, marker="o", color="red")

# optimalne portfolia pre jednotlive akcie
for i in range(len(tickers)):
    plt.scatter(np.sqrt(V[i,i]), r_mean[i], marker='x', color="blue")
    plt.text(np.sqrt(V[i,i]) + 0.005, r_mean[i], tickers[i], fontsize=9, ha='left', va='center')

# naivne portfolio
plt.scatter(sigma_naive, r_naive, marker='x', color="red")
plt.text(sigma_naive + 0.005, r_naive, "naivne", fontsize=9, ha='left', va='center')

plt.xlabel(r"$\sigma_p$ (volatilita portfólia)")
plt.ylabel(r"$\bar{r}_p$ (očakávaný výnos portfólia)")
plt.grid(True)
plt.legend()
plt.show()
