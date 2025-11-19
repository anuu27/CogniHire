# src/airf/report_generator.py
import os, re, json
from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt

TEMPLATE_PATH = os.path.join('data','templates','report_ultra_premium.tex')
OUTPUT_TEX = 'airf_report.tex'
RADAR_PNG = 'radar.png'
_LATEX_SUBS = {
    '\\\\': r'\\textbackslash{}',
    '%': r'\\%',
    '&': r'\\&',
    '$': r'\\$',
    '#': r'\\#',
    '_': r'\\_',
    '{': r'\\{',
    '}': r'\\}',
    '~': r'\\textasciitilde{}',
    '^': r'\\^{}'
}
def _latex_escape(s: str) -> str:
    if s is None:
        return ''
    s = str(s)
    pattern = re.compile('|'.join(re.escape(k) for k in _LATEX_SUBS.keys()))
    return pattern.sub(lambda m: _LATEX_SUBS[m.group(0)], s)

def generate_radar(aspect_labels, aspect_values, outpath='radar.png'):
    # simple wrapper - reuse matplotlib code to ensure radar present
    try:
        N = max(3, len(aspect_labels))
        labels = aspect_labels[:N]
        values = [float(v) for v in aspect_values[:N]]
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        vals = np.concatenate([np.array(values), np.array([values[0]])])
        angs = np.concatenate([angles, angles[:1]])
        fig = plt.figure(figsize=(3,3), dpi=180)
        fig.patch.set_facecolor('#0f0f0f')
        ax = fig.add_subplot(111, polar=True)
        ax.set_facecolor('#0f0f0f')
        ax.set_theta_offset(np.pi/2)
        ax.set_theta_direction(-1)
        ax.plot(angs, vals, linewidth=2, color='#50F0AA')
        ax.fill(angs, vals, color='#50F0AA', alpha=0.25)
        ax.set_thetagrids(np.degrees(angles), labels, color='#A0A0A0')
        ax.set_ylim(0, 10)
        ax.grid(color='#303030', linestyle=':', linewidth=0.6)
        for sp in ax.spines.values():
            sp.set_visible(False)
        plt.tight_layout(pad=0.5)
        fig.savefig(outpath, bbox_inches='tight', facecolor=fig.get_facecolor(), dpi=180)
        plt.close(fig)
    except Exception:
        pass

def _generate_per_question_bars(per_question: List[Dict[str,Any]]):
    return '% omitted per-question details for compact report\\n'

def generate(aggregated_scores: Dict[str,Any], cv_scores: Dict[str,Any], per_question: List[Dict[str,Any]]):
    if not os.path.exists(TEMPLATE_PATH):
        raise FileNotFoundError('Template missing: ' + TEMPLATE_PATH)
    with open(TEMPLATE_PATH, 'r', encoding='utf-8') as f:
        template = f.read()
    data_map = {
        'ROLE_NAME': aggregated_scores.get('role_name','Role'),
        'CANDIDATE': aggregated_scores.get('candidate','Candidate'),
        'CCI': f"{aggregated_scores.get('CCI',0.0):.2f}",
        'BADGE': aggregated_scores.get('badge',''),
        'TECH_AVG_10': f"{aggregated_scores.get('technical_summary',{}).get('technical_average',0.0):.2f}",
        'CV_RELEVANCE': f"{cv_scores.get('relevance',0.0):.2f}",
        'SIGMA': f"{aggregated_scores.get('confidence',0.0):.3f}",
        'S_CONCEPTUAL': f"{aggregated_scores.get('strength',{}).get('aspect_mean',[0,0,0,0,0])[0]*10:.2f}",
        'S_REASONING': f"{aggregated_scores.get('strength',{}).get('aspect_mean',[0,0,0,0,0])[1]*10:.2f}",
        'S_PRECISION': f"{aggregated_scores.get('strength',{}).get('aspect_mean',[0,0,0,0,0])[2]*10:.2f}",
        'S_CLARITY': f"{aggregated_scores.get('strength',{}).get('aspect_mean',[0,0,0,0,0])[3]*10:.2f}",
        'S_CREATIVITY': f"{aggregated_scores.get('strength',{}).get('aspect_mean',[0,0,0,0,0])[4]*10:.2f}",
        'TRAJECTORY': aggregated_scores.get('trajectory',{}).get('label','Stable'),
        'STRENGTH': aggregated_scores.get('summary_text',''),
        'RISK': aggregated_scores.get('recommendation',''),
        'RECOMMENDATION': aggregated_scores.get('recommendation',''),
        'SUMMARY_TEXT': aggregated_scores.get('summary_text',''),
        'PER_QUESTION_BARS': _generate_per_question_bars(per_question),
    }
    filled = template
    for k,v in data_map.items():
        filled = filled.replace('{{'+k+'}}', _latex_escape(str(v)))
    with open('airf_report.tex','w',encoding='utf-8') as fout:
        fout.write(filled)
    return 'airf_report.tex'
