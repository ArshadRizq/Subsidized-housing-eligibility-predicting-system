import sys
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

from fuzzy_logic.inference import fuzzy_mamdani, fuzzy_sugeno
from utils.metrics import ground_truth_score, confusion_matrix_manual
from utils.plots import (
    plot_membership_functions,
    plot_mamdani_defuzz,
    plot_sugeno_defuzz,
    plot_score_distribution,
    plot_scatter_comparison,
    plot_segment_comparison,
)
from dl_model.model_registry import load_all_models, predict_all, batch_predict_all

DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'data.csv')
FEATURE_NAMES = ['Income', 'Household_Size', 'Work_Experience', 'Age', 'Number_of_Dependents']

st.set_page_config(page_title='Fuzzy Eligibility - Rumah Subsidi', layout='wide')
st.title('Sistem Prediksi Eligibilitas Penerima Rumah Subsidi')
st.markdown('Implementasi **Fuzzy Logic (Mamdani & Sugeno)** + **Integrasi Machine Learning** — from scratch')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_metrics(gt_scores, gt_labels, pred_scores, pred_labels):
    acc = float(np.mean(pred_labels == gt_labels))
    mae = float(np.mean(np.abs(pred_scores - gt_scores)))
    rmse = float(np.sqrt(np.mean((pred_scores - gt_scores) ** 2)))
    return acc, mae, rmse


@st.cache_data
def compute_batch_evaluation():
    """Run all 10k rows through all methods. Cached indefinitely."""
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_NAMES].values.astype(float)

    gt_scores = np.array([ground_truth_score(*row) for row in X])
    gt_labels = np.array(['Layak' if s >= 60 else 'Tidak Layak' for s in gt_scores])

    # Fuzzy inference — row-by-row (unavoidable for sequential rule eval)
    m_scores, m_labels, s_scores, s_labels = [], [], [], []
    for row in X:
        inc, hh, we, age, dep = row
        ms, ml, _ = fuzzy_mamdani(inc, hh, we, age, dep)
        ss, sl, _ = fuzzy_sugeno(inc, hh, we, age, dep)
        m_scores.append(ms); m_labels.append(ml)
        s_scores.append(ss); s_labels.append(sl)

    m_scores = np.array(m_scores); m_labels = np.array(m_labels)
    s_scores = np.array(s_scores); s_labels = np.array(s_labels)

    # ML model inference — vectorized
    ml_results = batch_predict_all(X)

    return {
        'gt_scores': gt_scores, 'gt_labels': gt_labels,
        'm_scores': m_scores, 'm_labels': m_labels,
        's_scores': s_scores, 's_labels': s_labels,
        'ml_results': ml_results,
        'incomes': X[:, 0],
    }


def build_scores_dict(batch_data):
    """Build a {name: scores_array} dict ordered: Fuzzy first, then ML by accuracy."""
    d = batch_data
    ml_results = d.get('ml_results', {})

    # Compute accuracy for sorting
    def _acc(name, scores, labels):
        if scores is None:
            return -1
        return float(np.mean(labels == d['gt_labels']))

    ml_ranked = sorted(
        [(name, arr[0], arr[1]) for name, arr in ml_results.items()],
        key=lambda x: -_acc(x[0], x[1], x[2]),
    )

    scores_dict = {
        'Mamdani': d['m_scores'],
        'Sugeno': d['s_scores'],
    }
    for name, scores, _ in ml_ranked:
        scores_dict[name] = scores

    return scores_dict


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    'Predictor', 'Rule Base', 'Membership Functions', 'Defuzzification', 'Comparison', 'Batch Evaluation',
])

# ---- Tab 1: Predictor ---------------------------------------------------
with tab1:
    st.header('Predictor Eligibilitas')
    st.markdown('Masukkan 5 variabel rumah tangga untuk memprediksi kelayakan subsidi.')

    col1, col2 = st.columns(2)
    with col1:
        income = st.number_input('Income (USD)', min_value=0, max_value=10_000_000,
                                  value=65_000, step=10_000, format='%d')
        household = st.slider('Household Size', min_value=1, max_value=7, value=4)
        dependents = st.slider('Number of Dependents', min_value=0, max_value=5, value=2)
    with col2:
        work_exp = st.slider('Work Experience (tahun)', min_value=0, max_value=50, value=10)
        age = st.slider('Age (tahun)', min_value=18, max_value=70, value=35)

    if st.button('Predict', type='primary'):
        gt = ground_truth_score(income, household, work_exp, age, dependents)
        gt_label = 'Layak' if gt >= 60 else 'Tidak Layak'
        ms, ml, _ = fuzzy_mamdani(income, household, work_exp, age, dependents)
        ss, sl, _ = fuzzy_sugeno(income, household, work_exp, age, dependents)

        st.subheader('Fuzzy Logic (Core)')
        col_f1, col_f2, col_f3 = st.columns(3)
        col_f1.metric('Ground Truth', f'{gt:.0f}', gt_label)
        col_f2.metric('Mamdani', f'{ms:.2f}', ml)
        col_f3.metric('Sugeno', f'{ss:.2f}', sl)

        st.subheader('Integrasi Machine Learning (Bonus)')
        try:
            ml_preds = predict_all(income, household, work_exp, age, dependents)
            ml_cols = st.columns(len(ml_preds))
            for col, (name, (score, label)) in zip(ml_cols, ml_preds.items()):
                col.metric(name, f'{score:.2f}', label)
        except Exception as e:
            st.warning(f'Model ML belum dilatih. Jalankan `python dl_model/train_all.py` dulu. ({e})')

        st.info('**Threshold Layak:** Skor \u2265 55')
    else:
        st.info('Tekan tombol **Predict** untuk melihat hasil.')

# ---- Tab 2: Rule Base ---------------------------------------------------------
with tab2:
    st.header('Rule Base')
    st.markdown('33 aturan **IF-THEN** yang mendasari inferensi fuzzy. '
                'Setiap aturan terdiri dari 5 anteseden (variabel input) dan 1 konsekuen (output).')

    from fuzzy_logic.rules import RULES

    OUTPUT_ORDER = ['Tidak_Layak', 'Kurang_Layak', 'Cukup_Layak', 'Layak']
    OUTPUT_COLORS = {'Tidak_Layak': 'red', 'Kurang_Layak': 'orange', 'Cukup_Layak': 'gold', 'Layak': 'green'}
    OUTPUT_EMOJI = {'Tidak_Layak': '🔴', 'Kurang_Layak': '🟠', 'Cukup_Layak': '🟡', 'Layak': '🟢'}

    LABEL_MAP = {
        'Rendah': 'Rendah', 'Menengah_Bawah': 'Menengah Bawah', 'Menengah': 'Menengah', 'Tinggi': 'Tinggi',
        'Kecil': 'Kecil', 'Sedang': 'Sedang', 'Besar': 'Besar',
        'Baru': 'Baru', 'Cukup': 'Cukup', 'Berpengalaman': 'Berpengalaman',
        'Muda': 'Muda', 'Produktif': 'Produktif', 'Senior': 'Senior',
        'Sedikit': 'Sedikit', 'Banyak': 'Banyak',
        'Tidak_Layak': 'Tidak Layak', 'Kurang_Layak': 'Kurang Layak',
        'Cukup_Layak': 'Cukup Layak', 'Layak': 'Layak',
    }

    rows = []
    for i, (inc, hh, we, age, dep, out) in enumerate(RULES, 1):
        rows.append({
            '#': i,
            'Income': LABEL_MAP[inc],
            'Household': LABEL_MAP[hh],
            'Work Exp': LABEL_MAP[we],
            'Age': LABEL_MAP[age],
            'Dependents': LABEL_MAP[dep],
            'Output': LABEL_MAP[out],
        })

    rules_df = pd.DataFrame(rows)

    def color_output(val):
        for key, color in OUTPUT_COLORS.items():
            if LABEL_MAP[key] == val:
                return f'color: {color}; font-weight: bold'
        return ''

    st.dataframe(
        rules_df.style.map(color_output, subset=['Output']),
        hide_index=True,
        width='stretch',
        column_config={
            '#': st.column_config.NumberColumn('#', width='small'),
            'Income': st.column_config.TextColumn('Income', width='medium'),
            'Household': st.column_config.TextColumn('Household', width='medium'),
            'Work Exp': st.column_config.TextColumn('Work Exp', width='medium'),
            'Age': st.column_config.TextColumn('Age', width='medium'),
            'Dependents': st.column_config.TextColumn('Dependents', width='medium'),
            'Output': st.column_config.TextColumn('Output', width='medium'),
        },
    )

    with st.expander('Ringkasan Distribusi Output'):
        from collections import Counter
        counts = Counter(r[-1] for r in RULES)
        dist_df = pd.DataFrame([
            {'Output': f'{OUTPUT_EMOJI[k]} {LABEL_MAP[k]}', 'Jumlah Aturan': v,
             'Persentase': f'{v/len(RULES)*100:.1f}%'}
            for k, v in sorted(counts.items(), key=lambda x: OUTPUT_ORDER.index(x[0]))
        ])
        st.dataframe(dist_df, hide_index=True, width='stretch')

    with st.expander('Tentang Aturan'):
        st.markdown("""
        Aturan ditulis dalam bentuk **IF-THEN**:
        - **IF** `Income = X` **∧** `Household = Y` **∧** `Work Exp = Z` **∧** `Age = W` **∧** `Dependents = V`
        - **THEN** `Output = O`

        Semua anteseden dihubungkan dengan operator **AND (∧)**, sehingga kekuatan
        aturan (fire strength) ditentukan oleh nilai minimum dari seluruh keanggotaan anteseden.
        """)

# ---- Tab 3: Membership Functions -----------------------------------------
with tab3:
    st.header('Fungsi Keanggotaan')
    st.markdown('Visualisasi fungsi keanggotaan untuk setiap variabel linguistik.')
    st.pyplot(plot_membership_functions())

# ---- Tab 4: Defuzzification ---------------------------------------------
with tab4:
    st.header('Defuzzification Viewer')
    st.markdown('Masukkan nilai input dan lihat proses defuzzifikasi Mamdani & Sugeno.')

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        inc_d = st.number_input('Income', min_value=0, max_value=10_000_000, value=65_000,
                                 step=10_000, format='%d', key='def_inc')
        hh_d = st.slider('Household Size', 1, 7, 4, key='def_hh')
        dep_d = st.slider('Dependents', 0, 5, 2, key='def_dep')
    with col_d2:
        we_d = st.slider('Work Experience', 0, 50, 10, key='def_we')
        age_d = st.slider('Age', 18, 70, 35, key='def_age')

    if st.button('Show Defuzzification', type='primary', key='def_btn'):
        ms, ml, agg = fuzzy_mamdani(inc_d, hh_d, we_d, age_d, dep_d)
        ss, sl, firings = fuzzy_sugeno(inc_d, hh_d, we_d, age_d, dep_d)

        col_f1, col_f2 = st.columns(2)
        with col_f1:
            st.pyplot(plot_mamdani_defuzz(agg, ms, ml, 'Mamdani - Centroid of Area'))
        with col_f2:
            st.pyplot(plot_sugeno_defuzz(firings, ss, sl, 'Sugeno - Weighted Average'))
    else:
        st.info('Tekan tombol **Show Defuzzification** untuk melihat grafik.')

# ---- Tab 5: Comparison --------------------------------------------------
with tab5:
    st.header('Perbandingan Semua Metode')
    st.markdown('Analisis perbandingan hasil prediksi **Fuzzy Logic (Core)** vs **Integrasi Machine Learning (Bonus)**.')

    if st.button('Run Comparison', type='primary', key='comp_btn'):
        with st.spinner('Menjalankan evaluasi batch...'):
            data = compute_batch_evaluation()
        scores_dict = build_scores_dict(data)
        st.pyplot(plot_score_distribution(scores_dict))
        st.pyplot(plot_scatter_comparison(scores_dict))
        st.pyplot(plot_segment_comparison(scores_dict, data['incomes']))
    else:
        st.info('Tekan tombol **Run Comparison** untuk memuat data dan grafik.')

# ---- Tab 6: Batch Evaluation --------------------------------------------
with tab6:
    st.header('Batch Evaluation')
    st.markdown('Evaluasi menyeluruh pada seluruh 10.000 baris dataset — **Fuzzy Logic (Core)** dan **Integrasi Machine Learning (Bonus)**.')

    if st.button('Run Batch Evaluation', type='primary', key='batch_btn'):
        with st.spinner('Running batch evaluation on 10,000 rows (this may take ~60s)...'):
            start = time.time()
            data = compute_batch_evaluation()
            elapsed = time.time() - start

        gt_labels = data['gt_labels']
        gt_scores = data['gt_scores']

        all_methods = []
        all_methods.append(('Mamdani', data['m_scores'], data['m_labels']))
        all_methods.append(('Sugeno', data['s_scores'], data['s_labels']))
        for name, (scores, labels) in data['ml_results'].items():
            all_methods.append((name, scores, labels))

        metrics_rows = []
        for name, scores, labels in all_methods:
            acc, mae, rmse = compute_metrics(gt_scores, gt_labels, scores, labels)
            metrics_rows.append({'Metode': name, 'Accuracy': acc, 'MAE': mae, 'RMSE': rmse})

        st.success(f'Batch evaluation completed in {elapsed:.1f}s')

        st.subheader('Perbandingan Metrik — Semua Metode')
        metrics_df = pd.DataFrame(metrics_rows)
        metrics_df = metrics_df.sort_values('Accuracy', ascending=False).reset_index(drop=True)
        metrics_df.index = metrics_df.index + 1

        def highlight_best(val, col_name):
            if col_name in ('Accuracy',):
                numeric_col = metrics_df[col_name]
            else:
                numeric_col = metrics_df[col_name]
            best = numeric_col.max() if col_name == 'Accuracy' else numeric_col.min()
            if val == best:
                return 'background-color: #90EE90; font-weight: bold'
            return ''

        st.dataframe(
            metrics_df.style.map(lambda v: '', subset=['Metode'])
            .apply(lambda s: [highlight_best(v, s.name) for v in s], subset=['Accuracy', 'MAE', 'RMSE']),
            hide_index=True,
            use_container_width=True,
        )

        st.subheader('Agreement Antar Metode (% label agreement)')
        method_names = [m[0] for m in all_methods]
        n = len(method_names)
        agree = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                v = float(np.mean(all_methods[i][2] == all_methods[j][2]))
                agree[i][j] = v
                agree[j][i] = v

        fig_agree, ax_agree = plt.subplots(figsize=(10, 8))
        im = ax_agree.imshow(agree, cmap='YlGn', vmin=0.5, vmax=1.0)
        ax_agree.set_xticks(range(n))
        ax_agree.set_yticks(range(n))
        ax_agree.set_xticklabels(method_names, fontsize=8, rotation=45, ha='right')
        ax_agree.set_yticklabels(method_names, fontsize=8)
        plt.colorbar(im, ax=ax_agree, label='Agreement')
        for i in range(n):
            for j in range(n):
                color = 'white' if agree[i][j] > 0.75 else 'black'
                ax_agree.text(j, i, f'{agree[i][j]:.3f}', ha='center', va='center',
                              fontsize=7, color=color)
        ax_agree.set_title('Pairwise Label Agreement', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig_agree)

        st.subheader('Confusion Matrices — Top 3 Metode vs Ground Truth')
        top3 = sorted(metrics_rows, key=lambda r: -r['Accuracy'])[:3]
        fig_cm, axes_cm = plt.subplots(1, 3, figsize=(15, 4))
        fig_cm.suptitle('Confusion Matrix: Top 3 Metode', fontsize=13, fontweight='bold')
        cmap_list = ['Blues', 'Greens', 'Oranges']
        for ax, row, cmap in zip(axes_cm, top3, cmap_list):
            method_scores, method_labels = None, None
            for name, scores, labels in all_methods:
                if name == row['Metode']:
                    method_labels = labels
                    break
            cm, classes = confusion_matrix_manual(gt_labels, method_labels)
            im = ax.imshow(cm, cmap=cmap)
            ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
            ax.set_xticklabels(classes); ax.set_yticklabels(classes)
            ax.set_xlabel('Prediksi'); ax.set_ylabel('Ground Truth')
            ax.set_title(f"{row['Metode']}  (Acc: {row['Accuracy']:.4f})", fontweight='bold')
            for i in range(2):
                for j in range(2):
                    color = 'white' if cm[i][j] > cm.max() / 2 else 'black'
                    ax.text(j, i, str(cm[i][j]), ha='center', va='center', fontsize=14, color=color)
        st.pyplot(fig_cm)

        st.subheader('Distribusi Skor — Semua Metode')
        scores_dict = build_scores_dict(data)
        st.pyplot(plot_score_distribution(scores_dict))

        st.subheader('Scatter Comparison')
        st.pyplot(plot_scatter_comparison(scores_dict))

        st.subheader('Rata-rata Skor per Segmen Income')
        st.pyplot(plot_segment_comparison(scores_dict, data['incomes']))
    else:
        st.info('Tekan tombol **Run Batch Evaluation** untuk memuat hasil evaluasi.')
