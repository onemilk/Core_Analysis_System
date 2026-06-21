"""ReportGenerator — Jinja2-based HTML report generation with matplotlib charts."""

import io
import base64
import os
from jinja2 import Environment, FileSystemLoader
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


class ReportGenerator:
    _template_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates")
    _env = Environment(loader=FileSystemLoader(_template_dir))

    @classmethod
    def generate_hole_report(cls, summary: dict, fill_stats: list,
                             effect: dict, info: dict) -> str:
        charts = cls._generate_hole_charts(summary.get("diameters", []))
        size_dist = cls._build_size_dist_table(summary.get("size_distribution", {}),
                                               summary.get("total_count", 0))
        template = cls._env.get_template("hole_report.html")
        html = template.render(
            info=info,
            summary={
                "total_count": summary["total_count"],
                "total_area_mm2": summary["total_area_mm2"],
                "avg_area_mm2": summary["avg_area_mm2"],
                "porosity_percent": summary["porosity_percent"],
                "avg_d_mm": summary["avg_equivalent_d_mm"],
                "max_d_mm": summary["max_equivalent_d_mm"],
                "min_d_mm": summary["min_equivalent_d_mm"],
            },
            fill_stats=fill_stats,
            effect=effect,
            size_dist=size_dist,
            charts=charts
        )
        return html

    @classmethod
    def generate_fracture_report(cls, summary: dict, fractures: list,
                                 type_stats: list, info: dict) -> str:
        template = cls._env.get_template("fracture_report.html")
        html = template.render(info=info, summary=summary,
                               fractures=fractures, type_stats=type_stats)
        return html

    @classmethod
    def _generate_hole_charts(cls, diameters: list) -> dict:
        if not diameters:
            return {"histogram": "", "cumulative": "", "normal_cdf": ""}
        diameters = np.array(diameters)
        charts = {}

        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        ax.hist(diameters, bins=min(10, len(diameters)), edgecolor='black', alpha=0.7)
        ax.set_xlabel("等效直径 (mm)", fontsize=12)
        ax.set_ylabel("频数", fontsize=12)
        ax.set_title("孔洞等效直径频率直方图", fontsize=14)
        ax.tick_params(labelsize=10)
        charts["histogram"] = cls._fig_to_b64(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        sorted_d = np.sort(diameters)
        cumulative = np.arange(1, len(sorted_d) + 1) / len(sorted_d) * 100
        ax.plot(sorted_d, cumulative, 'b-o', markersize=5, linewidth=1.5)
        ax.set_xlabel("等效直径 (mm)", fontsize=12)
        ax.set_ylabel("累计频率 (%)", fontsize=12)
        ax.set_title("孔洞等效直径累计频率曲线", fontsize=14)
        ax.tick_params(labelsize=10)
        charts["cumulative"] = cls._fig_to_b64(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        mu, sigma = np.mean(diameters), np.std(diameters)
        if sigma > 0:
            x = np.linspace(max(0, mu - 3 * sigma), mu + 3 * sigma, 100)
            ax.plot(x, stats.norm.cdf(x, mu, sigma) * 100, 'r-', lw=2, label=f"μ={mu:.2f}, σ={sigma:.2f}")
            ax.scatter(sorted_d, cumulative, s=15, alpha=0.5, label="实测")
            ax.legend(fontsize=10)
        ax.set_xlabel("等效直径 (mm)", fontsize=12)
        ax.set_ylabel("累计概率 (%)", fontsize=12)
        ax.set_title("孔洞等效直径正态累计曲线", fontsize=14)
        ax.tick_params(labelsize=10)
        charts["normal_cdf"] = cls._fig_to_b64(fig)
        plt.close(fig)
        return charts

    @staticmethod
    def _fig_to_b64(fig) -> str:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=200, bbox_inches='tight')
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()

    @staticmethod
    def _build_size_dist_table(size_dist: dict, total: int) -> list:
        ranges = {"大洞": ">10mm", "中洞": "5-10mm", "小洞": "1-4.9mm", "针孔/溶孔": "<1mm"}
        return [{"category": k, "range": ranges.get(k, ""),
                 "count": size_dist.get(k, 0),
                 "percent": round(size_dist.get(k, 0) / total * 100, 1) if total > 0 else 0}
                for k in ("大洞", "中洞", "小洞", "针孔/溶孔")]

    @classmethod
    def generate_grain_report(cls, summary: dict, info: dict) -> str:
        charts = cls._generate_grain_charts(summary.get("diameters", []),
                                            summary.get("feret_data", []))
        size_dist = cls._build_grain_size_table(
            summary.get("size_distribution", {}),
            summary.get("total_count", 0))
        template = cls._env.get_template("grain_report.html")
        html = template.render(
            info=info,
            summary={
                "total_count": summary["total_count"],
                "avg_d_mm": summary["avg_diameter_mm"],
                "md_mm": summary["md_diameter_mm"],
                "std_mm": summary["std_dev_mm"],
                "max_mm": summary["max_diameter_mm"],
                "min_mm": summary["min_diameter_mm"],
            },
            size_dist=size_dist,
            charts=charts
        )
        return html

    @classmethod
    def _generate_grain_charts(cls, diameters: list, feret_data: list) -> dict:
        if not diameters:
            return {"histogram": "", "cumulative": "", "feret": ""}
        diameters = np.array(diameters)
        charts = {}

        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        ax.hist(diameters, bins=min(12, len(diameters)), edgecolor='black', alpha=0.7)
        ax.set_xlabel("粒径 (mm)", fontsize=12)
        ax.set_ylabel("频数", fontsize=12)
        ax.set_title("粒度频率直方图", fontsize=14)
        ax.tick_params(labelsize=10)
        charts["histogram"] = cls._fig_to_b64(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        sorted_d = np.sort(diameters)
        cumulative = np.arange(1, len(sorted_d) + 1) / len(sorted_d) * 100
        ax.plot(sorted_d, cumulative, 'b-o', markersize=4, linewidth=1.5)
        ax.set_xlabel("粒径 (mm)", fontsize=12)
        ax.set_ylabel("累计频率 (%)", fontsize=12)
        ax.set_title("粒度累计频率曲线", fontsize=14)
        ax.tick_params(labelsize=10)
        charts["cumulative"] = cls._fig_to_b64(fig)
        plt.close(fig)

        if feret_data:
            fig, ax = plt.subplots(figsize=(7.5, 4.5))
            longs = [f[0] for f in feret_data]
            shorts = [f[1] for f in feret_data]
            ax.scatter(shorts, longs, alpha=0.6, s=20)
            max_val = max(max(longs), max(shorts)) * 1.1
            ax.plot([0, max_val], [0, max_val], 'r--', lw=1, alpha=0.5)
            ax.set_xlabel("Feret 短轴 (mm)", fontsize=12)
            ax.set_ylabel("Feret 长轴 (mm)", fontsize=12)
            ax.set_title("Feret 长轴-短轴散点图", fontsize=14)
            ax.tick_params(labelsize=10)
            charts["feret"] = cls._fig_to_b64(fig)
            plt.close(fig)
        else:
            charts["feret"] = ""

        return charts

    @staticmethod
    def _build_grain_size_table(size_dist: dict, total: int) -> list:
        ranges = {"砾": ">2mm", "砂": "0.0625-2mm", "粉砂": "0.0039-0.0625mm", "泥": "<0.0039mm"}
        return [{"category": k, "range": ranges.get(k, ""),
                 "count": size_dist.get(k, 0),
                 "percent": round(size_dist.get(k, 0) / total * 100, 1) if total > 0 else 0}
                for k in ("砾", "砂", "粉砂", "泥")]
