import numpy as np

def trimf(x, a, b, c):
    if x <= a or x >= c:
        return 0.0
    elif x <= b:
        return (x - a) / (b - a)
    else:
        return (c - x) / (c - b)

def trapmf(x, a, b, c, d):
    if x < a or x > d:
        return 0.0
    elif x <= b:
        return (x - a) / (b - a) if b != a else 1.0
    elif x <= c:
        return 1.0
    else:
        return (d - x) / (d - c) if d != c else 1.0

INCOME_CAP = 1_000_000

def mf_income(x):
    x = min(x, INCOME_CAP)
    return {
        'Rendah'         : trapmf(x, 0, 0, 70_000, 100_000),
        'Menengah_Bawah' : trimf(x, 70_000, 130_000, 250_000),
        'Menengah'       : trimf(x, 200_000, 400_000, 650_000),
        'Tinggi'         : trapmf(x, 500_000, 800_000, INCOME_CAP, INCOME_CAP),
    }

def mf_household(x):
    return {
        'Kecil'  : trapmf(x, 1, 1, 3, 4),
        'Sedang' : trapmf(x, 2, 3, 5, 6),
        'Besar'  : trapmf(x, 4, 5, 7, 7),
    }

def mf_workexp(x):
    return {
        'Baru'          : trapmf(x, 0, 0, 2, 5),
        'Cukup'         : trapmf(x, 3, 8, 18, 25),
        'Berpengalaman' : trapmf(x, 18, 25, 50, 50),
    }

def mf_age(x):
    return {
        'Muda'      : trapmf(x, 18, 18, 28, 38),
        'Produktif' : trapmf(x, 25, 35, 55, 65),
        'Senior'    : trapmf(x, 55, 65, 70, 70),
    }

def mf_dependents(x):
    return {
        'Sedikit' : trapmf(x, 0, 0, 1, 2),
        'Sedang'  : trimf(x, 1, 2, 4),
        'Banyak'  : trapmf(x, 3, 4, 5, 5),
    }

def mf_output(x):
    return {
        'Tidak_Layak'  : trapmf(x, 0, 0, 15, 30),
        'Kurang_Layak' : trimf(x, 20, 35, 55),
        'Cukup_Layak'  : trimf(x, 45, 62, 78),
        'Layak'        : trapmf(x, 68, 82, 100, 100),
    }
