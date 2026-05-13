#!/usr/bin/env python3
"""Generate Chinese PPT flowcharts as SVG files."""
import argparse
from pathlib import Path


FONT = "'Microsoft YaHei','SimHei','Noto Sans CJK SC',Arial,sans-serif"


def svg_header(width, height):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<defs>
  <marker id="arrow" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto" markerUnits="strokeWidth">
    <path d="M2,2 L10,6 L2,10 z" fill="#3b4652"/>
  </marker>
  <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
    <feDropShadow dx="0" dy="3" stdDeviation="4" flood-color="#102030" flood-opacity="0.16"/>
  </filter>
</defs>
<rect width="{width}" height="{height}" fill="#ffffff"/>
'''


def svg_footer():
    return "</svg>\n"


def text(x, y, content, size=28, color="#1f2933", weight=500, anchor="middle"):
    lines = str(content).split("\n")
    out = []
    line_height = size * 1.25
    start_y = y - (len(lines) - 1) * line_height / 2
    for i, line in enumerate(lines):
        out.append(
            f'<text x="{x}" y="{start_y + i * line_height}" text-anchor="{anchor}" '
            f'font-family="{FONT}" font-size="{size}" font-weight="{weight}" fill="{color}">{line}</text>'
        )
    return "\n".join(out)


def box(x, y, w, h, title, subtitle=None, fill="#f3f7fb", stroke="#2f6f8f"):
    cx, cy = x + w / 2, y + h / 2
    out = [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="3" filter="url(#shadow)"/>',
        text(cx, cy - (14 if subtitle else 0), title, size=28, weight=700),
    ]
    if subtitle:
        out.append(text(cx, cy + 30, subtitle, size=20, color="#506070", weight=400))
    return "\n".join(out)


def small_box(x, y, w, h, title, fill="#f7fafc", stroke="#8795a1"):
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="2"/>'
        + text(x + w / 2, y + h / 2, title, size=22, weight=600)
    )


def arrow(x1, y1, x2, y2, label=None):
    out = [
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
        f'stroke="#3b4652" stroke-width="4" marker-end="url(#arrow)"/>'
    ]
    if label:
        out.append(text((x1 + x2) / 2, (y1 + y2) / 2 - 12, label, size=18, color="#506070"))
    return "\n".join(out)


def title_block(title, subtitle=None, width=1600):
    out = [
        text(width / 2, 62, title, size=38, weight=800, color="#17212b"),
    ]
    if subtitle:
        out.append(text(width / 2, 108, subtitle, size=22, color="#53616f", weight=400))
    return "\n".join(out)


def save(path, content, width=1600, height=900):
    path.write_text(svg_header(width, height) + content + svg_footer(), encoding="utf-8")


def research_goal(out_dir):
    width, height = 1600, 900
    parts = [title_block("研究目标：从 Benchmark 到算法选择", "把全局性能比较转化为不同异常场景下的算法推荐规则")]
    xs = [90, 430, 770, 1110]
    labels = [
        ("ESA-ADB\nMission1", "卫星遥测异常检测数据"),
        ("官方经典算法", "PCC / HBOS / STD / iForest / KNN"),
        ("异常切片评估", "按异常类型、长度、局部性等分组"),
        ("算法选择规则", "Target_57 / Subset_6 分场景推荐"),
    ]
    for x, (t, s) in zip(xs, labels):
        parts.append(box(x, 300, 250, 170, t, s))
    for x in [340, 680, 1020]:
        parts.append(arrow(x, 385, x + 90, 385))
    parts.append(
        text(
            width / 2,
            660,
            "核心问题：不是只问“哪个算法平均最好”，而是问“哪个异常场景该用哪个算法”。",
            size=30,
            weight=700,
            color="#2f6f8f",
        )
    )
    save(out_dir / "01_research_goal_pipeline.svg", "\n".join(parts), width, height)


def benchmark_pipeline(out_dir):
    width, height = 1600, 900
    parts = [title_block("ESA-ADB Lite Benchmark 流程", "代码从 CSV 到结果表的实际执行路径")]
    ys = [170, 290, 410, 530, 650]
    items = [
        ("预处理 CSV", "train/test 宽表，含 timestamp、通道值、is_anomaly_* 标签"),
        ("通道组选择", "Target_57 或 Subset_6"),
        ("算法训练与打分", "PCC / HBOS / STD3 / STD5 / iForest / Subsequence_IF / KNN"),
        ("整体指标", "Point_AUPR / Point_AUROC / F1 / VUS / time_s"),
        ("切片指标", "slice_metrics.csv 与可靠切片排名"),
    ]
    for y, (t, s) in zip(ys, items):
        parts.append(box(260, y, 520, 82, t, s, fill="#f4f8fb"))
    for y in [252, 372, 492, 612]:
        parts.append(arrow(520, y, 520, y + 38))
    parts.append(box(930, 250, 390, 120, "run_benchmark.py", "主入口：循环 split / 通道组 / 算法", fill="#eef8f2", stroke="#4d8b31"))
    parts.append(box(930, 420, 390, 120, "metrics.py", "计算点级、通道均值、VUS 指标", fill="#fff7ed", stroke="#b86b25"))
    parts.append(box(930, 590, 390, 120, "slice_metrics.py", "按异常事件属性生成切片指标", fill="#f7f2ff", stroke="#8a5fbf"))
    parts.append(arrow(780, 450, 930, 310))
    parts.append(arrow(780, 500, 930, 480))
    parts.append(arrow(780, 610, 930, 650))
    save(out_dir / "02_lite_benchmark_pipeline.svg", "\n".join(parts), width, height)


def slice_definition(out_dir):
    width, height = 1600, 900
    parts = [title_block("异常切片如何定义", "切片来自官方异常标签和我们派生的事件统计特征")]
    parts.append(box(80, 180, 360, 110, "官方标签", "anomaly_types.csv", fill="#eef6ff", stroke="#2f6f8f"))
    parts.append(small_box(90, 340, 160, 64, "Category\nAnomaly/Rare"))
    parts.append(small_box(270, 340, 160, 64, "Length\nPoint/Subseq"))
    parts.append(small_box(90, 430, 160, 64, "Locality\nGlobal/Local"))
    parts.append(small_box(270, 430, 160, 64, "Dimensionality\nMulti/Uni"))
    parts.append(small_box(90, 520, 160, 64, "Class"))
    parts.append(small_box(270, 520, 160, 64, "Subclass"))

    parts.append(box(620, 180, 360, 110, "事件统计", "labels.csv", fill="#eef8f2", stroke="#4d8b31"))
    parts.append(small_box(640, 360, 145, 70, "Duration\n持续时间"))
    parts.append(small_box(815, 360, 145, 70, "ChannelCount\n影响通道数"))
    parts.append(small_box(640, 480, 145, 70, "Duration\nBucket"))
    parts.append(small_box(815, 480, 145, 70, "ChannelCount\nBucket"))
    parts.append(arrow(712, 430, 712, 480))
    parts.append(arrow(887, 430, 887, 480))

    parts.append(box(1150, 180, 340, 110, "通道组", "代码中固定定义", fill="#fff7ed", stroke="#b86b25"))
    parts.append(small_box(1180, 365, 125, 70, "Target_57\n主任务"))
    parts.append(small_box(1335, 365, 125, 70, "Subset_6\n局部子集"))
    parts.append(text(800, 710, "每个切片 = 一个通道组 + 一个异常属性取值", size=32, weight=800, color="#2f6f8f"))
    parts.append(text(800, 770, "例：Target_57 / Rare Event，Subset_6 / Local，Target_57 / Duration > 7d", size=24, color="#506070"))
    save(out_dir / "03_slice_definition.svg", "\n".join(parts), width, height)


def slice_evaluation(out_dir):
    width, height = 1600, 900
    parts = [title_block("切片评价与 Top-k 排名", "Top-k 表示算法在切片中的排名，不是预测点的 Top-k")]
    xs = [90, 370, 650, 930, 1210]
    labels = [
        ("选一个切片", "如 Target_57 / Rare Event"),
        ("筛选事件", "event_count >= 3 才进主结论"),
        ("构造标签", "切片事件为正样本，正常点为负样本"),
        ("计算指标", "Point_AUPR 为主，Point_AUROC 辅助"),
        ("算法排序", "统计 Top-1 / Top-2 / Top-3"),
    ]
    for x, (t, s) in zip(xs, labels):
        parts.append(box(x, 260, 230, 170, t, s))
    for x in [320, 600, 880, 1160]:
        parts.append(arrow(x, 345, x + 50, 345))
    parts.append(box(210, 580, 360, 120, "Top-1", "该算法在某个切片中排名第 1", fill="#eef8f2", stroke="#4d8b31"))
    parts.append(box(620, 580, 360, 120, "Top-2 / Top-3", "该算法在某个切片中进入前 2 / 前 3", fill="#fff7ed", stroke="#b86b25"))
    parts.append(box(1030, 580, 360, 120, "胜率统计", "看算法在可靠切片中有多稳定", fill="#f7f2ff", stroke="#8a5fbf"))
    parts.append(text(800, 790, "输出文件：slice_metrics.csv → slice_algorithm_ranking_reliable.csv → algorithm_win_counts_reliable.csv", size=22, color="#506070"))
    save(out_dir / "04_slice_evaluation_topk.svg", "\n".join(parts), width, height)


def recommendation(out_dir):
    width, height = 1600, 900
    parts = [title_block("ESA-ADB Mission1 算法选择规则", "从切片胜率和计算成本得到的推荐策略")]
    parts.append(box(90, 180, 360, 140, "Target_57", "默认主检测器：HBOS\n依据：Top-1 = 15/20", fill="#eef6ff", stroke="#2f6f8f"))
    parts.append(box(90, 390, 360, 140, "Target_57 Local", "当前薄弱场景\n后续重点优化", fill="#fff1f2", stroke="#b94444"))
    parts.append(box(620, 180, 360, 140, "Subset_6", "多算法候选\niForest / Subsequence_IF / KNN", fill="#eef8f2", stroke="#4d8b31"))
    parts.append(box(620, 390, 360, 140, "Local / Subsequence", "定向使用 Subsequence_IF\n不做全量默认", fill="#f7f2ff", stroke="#8a5fbf"))
    parts.append(box(1150, 180, 360, 140, "辅助候选", "KNN\n经常进入 Top-2 / Top-3", fill="#fff7ed", stroke="#b86b25"))
    parts.append(box(1150, 390, 360, 140, "快速筛查", "HBOS / PCC\n成本低，适合初筛", fill="#f8fafc", stroke="#64748b"))
    parts.append(text(800, 680, "最终结论：把 benchmark 结果转化为场景化算法选择，而不是只给一个全局排行榜。", size=30, weight=800, color="#2f6f8f"))
    save(out_dir / "05_algorithm_recommendation.svg", "\n".join(parts), width, height)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Chinese PPT flowcharts")
    parser.add_argument(
        "--output-dir",
        default="results/autodl/20260512_235913/ppt_flowcharts",
        help="Output directory for SVG flowcharts",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    research_goal(out_dir)
    benchmark_pipeline(out_dir)
    slice_definition(out_dir)
    slice_evaluation(out_dir)
    recommendation(out_dir)
    print(f"Wrote PPT flowcharts to {out_dir}")


if __name__ == "__main__":
    main()
