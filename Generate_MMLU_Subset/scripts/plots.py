import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def try_import_seaborn():
    try:
        import seaborn as sns  # noqa: F401
        return True
    except Exception:
        return False


def generate_charts(per_seed_csv: str, aggregate_csv: str, summary_json: str, out_dir: str = 'subset_analysis_charts'):
    HAS_SNS = try_import_seaborn()
    os.makedirs(out_dir, exist_ok=True)

    per_seed_df = pd.read_csv(per_seed_csv)
    agg_df = pd.read_csv(aggregate_csv)
    summary = None
    if os.path.exists(summary_json):
        with open(summary_json, 'r', encoding='utf-8') as f:
            summary = json.load(f)

    method_order = ['BUCKET', 'MATRIX', 'IRT']
    per_seed_df['方法'] = pd.Categorical(per_seed_df['方法'], categories=method_order, ordered=True)
    agg_df['方法'] = pd.Categorical(agg_df['方法'], categories=method_order, ordered=True)

    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 1) 胜出次数
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    plot_df = agg_df.sort_values('方法')
    x = np.arange(len(plot_df))
    vals = plot_df['胜出次数/10'].values
    if HAS_SNS:
        import seaborn as sns
        sns.barplot(x='方法', y='胜出次数/10', data=plot_df, ax=ax, palette='Set2')
    else:
        ax.bar(x, vals, color=['#66c2a5', '#8da0cb', '#fc8d62'])
        ax.set_xticks(x)
        ax.set_xticklabels(plot_df['方法'])
    ax.set_title('各方法胜出次数（10个random_state）')
    ax.set_xlabel('方法')
    ax.set_ylabel('胜出次数/10')
    for i, v in enumerate(vals):
        ax.text(i, v + 0.1, str(int(v)), ha='center', va='bottom', fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'wins_per_method.png'))
    plt.close(fig)

    # 2) 平均排名（误差棒）
    rank_stats = per_seed_df.groupby('方法')['排名'].agg(['mean', 'std']).reindex(method_order)
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.errorbar(rank_stats.index, rank_stats['mean'], yerr=rank_stats['std'], fmt='o-', capsize=4)
    ax.set_title('平均排名（误差棒为标准差）')
    ax.set_xlabel('方法')
    ax.set_ylabel('排名（越小越好）')
    for i, (m, row) in enumerate(rank_stats.iterrows()):
        ax.text(i, row['mean'], f"{row['mean']:.2f}", ha='center', va='bottom', fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'avg_rank_with_std.png'))
    plt.close(fig)

    # 3) 总分分布（箱线图）
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    if HAS_SNS:
        import seaborn as sns
        sns.boxplot(x='方法', y='总分', data=per_seed_df, ax=ax, palette='Set3')
        sns.stripplot(x='方法', y='总分', data=per_seed_df, ax=ax, color='k', alpha=0.5, size=4, jitter=True)
    else:
        data_list = [per_seed_df[per_seed_df['方法'] == m]['总分'].values for m in method_order]
        ax.boxplot(data_list, labels=method_order, patch_artist=True, boxprops=dict(facecolor='#d9d9d9'))
        ax.scatter(per_seed_df['方法'].cat.codes + 1, per_seed_df['总分'], s=10, c='k', alpha=0.6)
    ax.set_title('各方法总分分布（10个random_state）')
    ax.set_xlabel('方法')
    ax.set_ylabel('总分（越高越好）')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'total_score_box.png'))
    plt.close(fig)

    # 4) 代表性散点图
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    if HAS_SNS:
        import seaborn as sns
        sns.scatterplot(
            data=per_seed_df, x='与Full均值差(均值)', y='与Full相关(均值)', hue='方法', hue_order=method_order, style='方法', ax=ax, s=60, palette='Set2'
        )
    else:
        colors = {'BUCKET': '#66c2a5', 'MATRIX': '#8da0cb', 'IRT': '#fc8d62'}
        for m in method_order:
            dfm = per_seed_df[per_seed_df['方法'] == m]
            ax.scatter(dfm['与Full均值差(均值)'], dfm['与Full相关(均值)'], s=60, c=colors[m], label=m)
    ax.set_title('代表性：与完整MMLU相关 vs 均值差')
    ax.set_xlabel('与Full均值差（越小越好）')
    ax.set_ylabel('与Full相关（越大越好）')
    ax.legend(title='方法')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'representativeness_scatter.png'))
    plt.close(fig)

    # 5) 一致性散点图
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    if HAS_SNS:
        import seaborn as sns
        sns.scatterplot(
            data=per_seed_df, x='两子集模型差(均值)', y='两子集相关', hue='方法', hue_order=method_order, style='方法', ax=ax, s=60, palette='Set1'
        )
    else:
        colors = {'BUCKET': '#e41a1c', 'MATRIX': '#377eb8', 'IRT': '#4daf4a'}
        for m in method_order:
            dfm = per_seed_df[per_seed_df['方法'] == m]
            ax.scatter(dfm['两子集模型差(均值)'], dfm['两子集相关'], s=60, c=colors[m], label=m)
    ax.set_title('一致性：两子集相关 vs 模型差(均值)')
    ax.set_xlabel('两子集模型差(均值)（越小越好）')
    ax.set_ylabel('两子集相关（越大越好）')
    ax.legend(title='方法')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'consistency_scatter.png'))
    plt.close(fig)

    # 6) 排名热力图
    rank_pivot = per_seed_df.pivot_table(index='seed', columns='方法', values='排名')
    fig, ax = plt.subplots(figsize=(6, max(3, 0.4 * len(rank_pivot))), dpi=150)
    if HAS_SNS:
        import seaborn as sns
        sns.heatmap(rank_pivot.loc[sorted(rank_pivot.index)], annot=True, fmt='.0f', cmap='YlGnBu', cbar=True, ax=ax)
    else:
        im = ax.imshow(rank_pivot.loc[sorted(rank_pivot.index)].values, cmap='YlGnBu', aspect='auto')
        ax.set_xticks(np.arange(len(rank_pivot.columns)))
        ax.set_xticklabels(rank_pivot.columns)
        ax.set_yticks(np.arange(len(rank_pivot.index)))
        ax.set_yticklabels(sorted(rank_pivot.index))
        for i in range(len(rank_pivot.index)):
            for j in range(len(rank_pivot.columns)):
                val = rank_pivot.loc[sorted(rank_pivot.index)[i], rank_pivot.columns[j]]
                ax.text(j, i, f'{int(val)}', ha='center', va='center', color='black', fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title('每个random_state的排名（数值越小越好）')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'per_seed_rank_heatmap.png'))
    plt.close(fig)

    # 7) 汇总柱状图
    key_cols = ['总分(均值)', '与Full相关(均值)', '与Full均值差(均值)', '两子集相关(均值)', '两子集均值差(均值)', '两子集模型差(均值)']
    agg_plot = agg_df.set_index('方法').loc[method_order, key_cols]

    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    agg_plot[['总分(均值)', '与Full相关(均值)', '两子集相关(均值)']].plot(kind='bar', ax=ax)
    ax.set_title('关键正向指标对比（越高越好）')
    ax.set_xlabel('方法')
    ax.set_ylabel('数值')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'aggregate_positive_metrics.png'))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    agg_plot[['与Full均值差(均值)', '两子集均值差(均值)', '两子集模型差(均值)']].plot(kind='bar', ax=ax)
    ax.set_title('关键反向指标对比（越低越好）')
    ax.set_xlabel('方法')
    ax.set_ylabel('数值')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'aggregate_negative_metrics.png'))
    plt.close(fig)


