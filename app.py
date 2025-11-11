from flask import Flask, render_template, request
import random

app = Flask(__name__)

# ====================================================
# TUGAS 1 - PENGENALAN SOFT COMPUTING
# ====================================================
@app.route("/")
def tugas1():
    data = {
        "judul": "Tugas 1 - Soft Computing",
        "pengertian": "Soft Computing adalah kumpulan teknik komputasi cerdas seperti Fuzzy Logic, Neural Network, dan Evolutionary Computation yang mampu menangani ketidakpastian, ambiguitas, dan ketidaktepatan secara efisien.",
        "fuzzy": "Logika fuzzy (Fuzzy Logic) merupakan metode pengambilan keputusan yang meniru cara berpikir manusia, di mana nilai tidak selalu 0 atau 1, tetapi bisa berada di antara keduanya.",
        "nn": "Neural Network adalah sistem pembelajaran mesin yang meniru jaringan saraf otak manusia untuk mempelajari pola dan melakukan prediksi secara otomatis."
    }
    return render_template("tugas1.html", **data)


# ====================================================
# TUGAS 2 - FUZZY SUGENO
# ====================================================
@app.route("/tugas2", methods=["GET", "POST"])
def tugas2():
    hasil = None

    if request.method == "POST":
        ipk = float(request.form["ipk"])
        sertifikat = float(request.form["sertifikat"])

        μ_ipk_rendah = max(0, (3 - ipk) / 3) if ipk <= 3 else 0
        μ_ipk_sedang = max(0, 1 - abs(ipk - 3) / 1)
        μ_ipk_tinggi = max(0, (ipk - 2.5) / 1.5) if ipk >= 2.5 else 0

        μ_sert_sedikit = max(0, (5 - sertifikat) / 5)
        μ_sert_cukup = max(0, 1 - abs(sertifikat - 5) / 3)
        μ_sert_banyak = max(0, (sertifikat - 3) / 7) if sertifikat >= 3 else 0

        z1, z2, z3 = 40, 70, 90

        w1 = min(μ_ipk_rendah, μ_sert_sedikit)
        w2 = min(μ_ipk_sedang, μ_sert_cukup)
        w3 = min(μ_ipk_tinggi, μ_sert_banyak)

        hasil_sugeno = (w1*z1 + w2*z2 + w3*z3) / (w1 + w2 + w3) if (w1+w2+w3) != 0 else 0

        if hasil_sugeno < 60:
            kelayakan = "Tidak Layak"
        elif hasil_sugeno < 80:
            kelayakan = "Perlu Dipertimbangkan"
        else:
            kelayakan = "Layak"

        hasil = {
            "ipk": ipk,
            "sertifikat": sertifikat,
            "μ_ipk_rendah": round(μ_ipk_rendah, 2),
            "μ_ipk_sedang": round(μ_ipk_sedang, 2),
            "μ_ipk_tinggi": round(μ_ipk_tinggi, 2),
            "μ_sert_sedikit": round(μ_sert_sedikit, 2),
            "μ_sert_cukup": round(μ_sert_cukup, 2),
            "μ_sert_banyak": round(μ_sert_banyak, 2),
            "w1": round(w1, 2),
            "w2": round(w2, 2),
            "w3": round(w3, 2),
            "output_nilai": round(hasil_sugeno, 2),
            "hasil": kelayakan
        }

    return render_template("tugas2.html", hasil=hasil)


# ====================================================
# TUGAS 3 - ALGORITMA GENETIKA (KNAPSACK)
# ====================================================
items = {
    'A': {'weight': 7, 'value': 5},
    'B': {'weight': 2, 'value': 4},
    'C': {'weight': 1, 'value': 7},
    'D': {'weight': 9, 'value': 2},
}
capacity = 15
item_list = list(items.keys())

# Fungsi decode kromosom
def decode(chromosome):
    total_weight = 0
    total_value = 0
    selected_items = []
    for i, bit in enumerate(chromosome):
        if bit == 1:
            item = item_list[i]
            total_weight += items[item]['weight']
            total_value += items[item]['value']
            selected_items.append(item)
    if total_weight > capacity:
        total_value = 0  # penalti
    return selected_items, total_weight, total_value

# Fitness
def fitness(chromosome):
    _, w, v = decode(chromosome)
    return v if w <= capacity else 0

# Seleksi
def selection(population):
    population.sort(key=lambda x: fitness(x), reverse=True)
    return population[:2]

# Crossover
def crossover(p1, p2):
    point = random.randint(1, len(p1) - 2)
    c1 = p1[:point] + p2[point:]
    c2 = p2[:point] + p1[point:]
    return c1, c2

# Mutasi
def mutate(chromosome, prob=0.1):
    for i in range(len(chromosome)):
        if random.random() < prob:
            chromosome[i] = 1 - chromosome[i]
    return chromosome

@app.route("/tugas3", methods=["GET", "POST"])
def tugas3():
    hasil = None
    if request.method == "POST":
        generations = int(request.form["generations"])
        pop_size = int(request.form["pop_size"])

        # Inisialisasi populasi
        population = [[random.randint(0, 1) for _ in range(len(items))] for _ in range(pop_size)]
        history = []

        # Evolusi
        for g in range(generations):
            population = sorted(population, key=lambda x: fitness(x), reverse=True)
            best = population[0]
            best_items, best_w, best_v = decode(best)
            history.append({
                "generation": g + 1,
                "chromosome": best,
                "items": best_items,
                "weight": best_w,
                "value": best_v,
                "fitness": fitness(best),
            })

            # Buat generasi berikutnya
            new_pop = []
            for _ in range(pop_size // 2):
                p1, p2 = selection(population)
                c1, c2 = crossover(p1, p2)
                new_pop.extend([mutate(c1[:]), mutate(c2[:])])
            population = new_pop

        # Hasil terbaik
        best_final = max(population, key=fitness)
        best_items, best_w, best_v = decode(best_final)
        hasil = {
            "history": history,
            "best": {
                "chromosome": best_final,
                "items": best_items,
                "weight": best_w,
                "value": best_v,
                "fitness": fitness(best_final),
            },
        }

    return render_template("tugas3.html", hasil=hasil)

if __name__ == "__main__":
    app.run(debug=True)
