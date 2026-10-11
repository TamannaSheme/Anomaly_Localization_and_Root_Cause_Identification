"""Build the measured Results section as PDF, HTML and editable Markdown."""
import base64
import csv
import html
import json
from pathlib import Path
import re
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"analysis"
R=OUT/"results"


def read_csv(name):
    with (R/name).open() as f:
        return list(csv.DictReader(f))


def main():
    demo=read_csv("table_7_1.csv"); aer=read_csv("table_7_2.csv"); smd=read_csv("table_7_3.csv")
    validation=json.loads((R/"validation.json").read_text())
    assert validation['passed']
    assert [int(x['correct']) for x in demo] == [0,8], "Revise dated narrative for changed results."
    assert all(float(x['ac1'])==1 and float(x['ac3'])==1 for x in aer)
    assert float(smd[0]['hit1'])==.625 and float(smd[0]['ndcg3'])==.75
    blocks=[]
    def h(text,level=2): blocks.append(("heading",text,level))
    def p(text): blocks.append(("paragraph",text))
    def table(title,headers,rows,widths): blocks.append(("table",title,headers,rows,widths))
    def figure(name,caption,width=468): blocks.append(("figure",name,caption,width))
    def page(): blocks.append(("page",))

    h("7. Preliminary Results and Implementation Progress",1)
    p("Semester-final empirical study | 14 September 2026")
    p("This section reports three completed stages: a reconstructed source-versus-symptom demonstration and propagation sweep, an execution reproduction of the official AERCA linear experiment, and a small SMD affected-variable localisation study. These stages answer different questions and are not pooled into a single benchmark.")
    h("7.1 Experimental Setup")
    p("Experiments ran on an arm64 macOS CPU with Python 3.9.6, NumPy 2.0.2, pandas 2.3.3, Matplotlib 3.9.4 and PyTorch 2.8.0. AERCA used one PyTorch CPU thread. The software snapshot, commands, raw case results, generated arrays and model checkpoints are retained in the analysis directory. Synthetic diagnostics required approximately three seconds; no neural training was needed for the demonstration or sweep.")
    p("The original synthetic_system.py and per-case results were unavailable. A new generator was reconstructed from the report's description, with its equations and settings documented explicitly. Consequently, the previous 0/30 magnitude and 17/30 onset figures remain historical report claims, not independently reproduced results. Table 7.1 below contains only newly measured values. The supplied Word report and original PNG have been preserved.")
    p("The 30 demonstration cases use seeds 0-29. The main sweep uses 10 paired seeds in each of 16 gain-delay conditions, yielding 160 condition-specific cases; a further 160 cases examine the quieter-downstream noise profile. Pairing reuses the same background noise across conditions. All synthetic scores use only each case's 100 pre-fault observations for the reference mean and sample standard deviation. Fault boundaries are supplied, so this is an oracle-event experiment rather than anomaly detection.")
    h("7.2 Controlled Source-versus-Symptom Demonstration")
    p("The reconstructed four-variable response chain is Pump -> Flow -> Pressure -> Temperature. The pump receives a persistent step of size 2.5 at time 100 in a 200-step series. Each link delays and multiplies the fault response. Demonstration gains are sampled independently from [1.15, 1.6), delays from integers 1-5, and background noise scales vary by up to 15% around (1.0, 0.8, 0.65, 0.5). These choices were fixed before inspecting outcomes; they were not fitted to recover the original accuracy.")
    table("Table 7.1. Newly measured source accuracy on 30 reconstructed cases.",
          ["Method","Correct / 30","Acc@1","95% Wilson interval","Most common wrong pick"],
          [[r['method'],f"{r['correct']} / 30",f"{float(r['acc1']):.1%}",f"{float(r['ci95_low']):.1%} to {float(r['ci95_high']):.1%}",r['most_common_wrong'].capitalize()] for r in demo],
          [86,61,51,113,157])
    p("Magnitude ranking selected the true source in 0/30 cases; onset ranking selected it in 8/30 (26.7%). The most common incorrect selections were temperature and flow, respectively. This supports a bounded claim: in this controlled configuration, stronger or earlier threshold-crossing downstream responses can mislead simple source-ranking rules. It does not establish that established RCA methods fail.")
    page()
    h("7.2 Controlled Demonstration: Mechanism and Example")
    figure("figure_7_1.png","Figure 7.1. First evaluated demonstration case (seed 0), selected by seed rather than performance. The shaded region begins at the supplied fault time. Top: observed signals. Bottom: absolute z-scores relative to the pre-fault reference. Both baselines select downstream variables in this case.")
    p("For variable j, the stationary background follows b_j(t) = 0.65 b_j(t-1) + e_j(t), with innovation standard deviation sigma_j sqrt(1 - 0.65^2). The initial background is sampled from its stationary distribution. The pump response is r_0(t) = 2.5 for t >= 100 and zero earlier. Downstream responses follow r_j(t) = g_j r_(j-1)(t-d_j), and observed signals are x_j(t) = b_j(t) + r_j(t).")
    p("Magnitude ranking chooses the largest post-fault maximum absolute z-score. Onset ranking chooses the earliest start of three consecutive samples with |z| >= 3.0; if no variable crosses, it falls back to magnitude. The original code's tie policy is retained: tied variables are resolved in column order (pump, flow, pressure, temperature).")
    p("This generator models propagation of the injected response, while its normal background processes are independent. The normal data therefore do not contain a learnable causal chain. It is a diagnostic demonstration, not a validated physical process model or a suitable normal-training dataset for causal discovery. AERCA is evaluated separately on its own structural dynamical system.")
    page()
    h("7.3 Effect of Propagation Strength and Delay")
    p("The main sweep fixes stationary background standard deviations to (1, 1, 1, 1), the source step to 2.5, and AR persistence to 0.65. Per-link gain g takes values 0.5, 1.0, 1.5 and 2.0; per-link delay d takes values 1, 3, 5 and 10 time steps. Temperature therefore receives a response of 2.5 g^3 after 3d steps. The longest delay still leaves 70 observed samples after the final downstream arrival.")
    figure("figure_7_2.png","Figure 7.2. Acc@1 across propagation conditions with equal stationary background noise. Each cell reports correct source predictions out of 10 paired seeds. The main comparison contains 160 condition-specific cases; rows and columns share the same seeds.")
    p("Magnitude ranking recovered the source in 9/10 cases at g = 0.5 and 4/10 at g = 1.0 for every tested delay. At g = 1.5 and g = 2.0, it recovered the source in 0/10 at every delay. Thus the observed magnitude failure region is associated with downstream amplification; changing delay alone did not change these cell accuracies.")
    p("Onset evidence behaved differently. At g = 1.0, accuracy rose from 5/10 at d = 1 to 8/10 at d = 10. At g = 1.5 it rose from 3/10 to 6/10, and at g = 2.0 from 3/10 to 5/10. Longer temporal separation helped in these paired cases, but did not remove the effect of a noisy source remaining below the sustained threshold.")
    p("Noise matters independently of raw gain. With the quieter-downstream profile (1.0, 0.8, 0.65, 0.5), magnitude accuracy at g = 1.0 fell from 4/10 to 0/10 at every delay. Raw attenuation therefore does not guarantee attenuation of standardized anomaly evidence. These sensitivity results are exported separately without adding another main figure.")
    p("Ten seeds per cell provide coarse estimates: a cell with 0/10 has a 95% Wilson interval of approximately 0-27.8%, and 9/10 has an interval of 59.6-98.2%. The saved CSV includes all cell intervals. Four primary-sweep cases had tied earliest onsets; in two, column-order tie breaking favoured the pump. Random resolution would reduce the affected cells by five percentage points each. No significance claim is made from this small paired grid.")
    page()
    h("7.4 Preliminary AERCA Baseline Reproduction")
    p("The official AERCA repository [1, 2] was run unchanged at commit c3dd49d88568d77c3e1ecb9a40472cd77a79b930. The supported linear system was used with seeds 0, 1 and 2. Each run generated 10 normal sequences for the official 8/2 training/validation split and 100 held-out anomalous test sequences, each 500 time steps long with four variables. The official architecture, learning rate, maximum of 5000 epochs, and early-stopping patience of 20 were retained. Execution used CPU instead of CUDA and the locally installed dependency versions.")
    p("This system has lagged structural dependencies during normal operation. Its default anomaly is a positive exogenous pulse of size 3 lasting one step, with noise standard deviation 0.4; between one and four variables are selected as injected sources. The setting is deliberately separate from the persistent single-source response-chain experiment. Across the three test sets, 82 of 300 cases had exactly one injected source.")
    table("Table 7.2. Official AERCA linear-system reproduction; 100 test sequences per seed.",
          ["Seed","Epochs / best","AC@1","AC@3","AC*@1","Causal AUROC"],
          [[r['seed'],f"{r['epochs_run']} / {r['best_epoch']}",f"{float(r['ac1']):.1%}",f"{float(r['ac3']):.1%}",f"{float(r['ac_star1']):.1%}",f"{float(r['causal_auroc']):.3f}"] for r in aer],
          [38,105,66,66,66,127])
    p("AERCA achieved AC@1 = 100% and AC@3 = 100% on each seed, with causal-discovery AUROC = 1.000. Training stopped after 791, 1114 and 1266 epochs; the best checkpoints occurred 20 epochs earlier. These are successful execution and metric-reproduction results on an easy supported synthetic setting, not evidence that AERCA outperforms the two diagnostics on the custom generator or achieves these values on real data.")
    p("Metric definitions follow the official code. At an annotated anomaly time, AC@K is the number of labelled sources among the top K ranked variables divided by min(K, number of labelled sources), then averaged over anomalous times and sequences. AC@1 is a top-one hit measure; AC@3 is not the usual 'any source in top three' Hit@3. Exported per-sequence scores were independently checked against both official AC metrics, including the two-step model alignment.")
    p("The supplementary official AC*@1 searches ranked time-variable entries after POT thresholding. Its thresholds are fitted using scores pooled across the test sequences; this is a transductive part of the upstream evaluation. It is reported with the star and kept separate from the oracle-time AC@K metrics. AC*@1 was 96%, 99% and 99% across the seeds.")
    p("Implementation limitation: this run preserves upstream code behaviour. The helper named compute_kl_divergence uses a positive log-determinant term, differing from the standard Gaussian KL expression. The implementation was not silently corrected. Accordingly, this block validates that the released code and evaluation run reproducibly; it does not certify mathematical equivalence to every equation or result in the paper.")
    page()
    h("7.5 Real-Data Pipeline Validation: SMD")
    p("One SMD machine, machine-1-1, was selected before scoring from the official OmniAnomaly release [3]. Its train and test sets each contain 28,479 rows and 38 dimensions. All eight interpretation-labelled intervals were evaluated, with mean and sample standard deviation fitted only on the training set. Each event's dimensions were ranked by the maximum absolute z-score within the annotated interval. These labels identify affected dimensions, not physical root causes.")
    table("Table 7.3. Event-macro affected-variable localisation on eight SMD intervals.",
          ["Method","Hit@1","Hit@3","NDCG@3","Recall@3"],
          [["Magnitude (std floor 1e-8)",f"{float(smd[0]['hit1']):.1%}",f"{float(smd[0]['hit3']):.1%}",f"{float(smd[0]['ndcg3']):.3f}",f"{float(smd[0]['recall3']):.1%}"],
           ["Uniform random (exact expectation)",f"{float(smd[1]['hit1']):.1%}",f"{float(smd[1]['hit3']):.1%}",f"{float(smd[1]['ndcg3']):.3f}",f"{float(smd[1]['recall3']):.1%}"]],
          [210,62,62,67,67])
    figure("figure_7_3.png","Figure 7.3. First SMD interval, [15849, 16368). Top: the three highest-ranked dimensions, on a symmetric-log axis (linear within +/-1). Bottom: top ten scores; blue bars indicate labelled dimensions. Labelled IDs are 1, 9, 10, 12, 13, 14 and 15. D29 is unlabelled and constant in training.",width=425)
    p("Hit@K means at least one labelled dimension occurs in the top K. NDCG@3 uses binary relevance; Recall@3 reports the fraction of labelled dimensions recovered. Macro averaging gives each interval equal weight. The random reference accounts for differing label-set sizes, including one broad event with 31 of 38 dimensions labelled; therefore Hit@3 alone is not a sufficient summary.")
    page()
    h("7.5 Real-Data Validation: Sensitivity and Label Audit")
    p("Eight dimensions are constant in training. With a standard-deviation floor of 1e-8, a tiny new value can dominate the ranking. D29 is ranked first but is unlabelled in three of eight events. This exposes preprocessing sensitivity; it does not establish whether D29 is a physical cause.")
    p("A disclosed numerical sensitivity check varied the standard-deviation floor while holding data, events and metric definitions fixed. Floors 0.001, 0.01 and 0.1 each produced Hit@1 = 8/8; NDCG@3 was 0.971, 1.000 and 1.000, respectively. These alternatives were investigated after observing the constant-channel issue, so they are sensitivity results rather than a newly selected test-set optimum. The original floor remains the primary table result.")
    p("Intervals use zero-based, end-exclusive slices. All eight starts and seven ends match binary anomaly runs; the first interpretation ends 27 samples earlier. Evaluation covers only the 2667 interpretation-labelled samples out of 2694 binary-anomalous samples. Dimension IDs are interpreted as one-based and mapped to zero-based columns; raw labels are retained for audit. No detector was evaluated.")
    h("7.6 Preliminary Findings and Limits")
    p("First, the reconstructed controlled system demonstrates that downstream response amplification can make the strongest standardized anomaly a misleading source indicator. Second, temporal separation can improve onset-based source selection, although source noise, sustained-crossing thresholds and tied onsets remain relevant. Third, the noise profile changes the failure region, so raw propagation gain alone does not explain standardized rankings. Fourth, the official AERCA runs and SMD case study establish working model-reproduction, label-audit and metric-export pipelines.")
    p("These results remain preliminary. The custom generator uses independent normal backgrounds and always injects the source at the pump; an identity-only 'always pump' rule would score 100%, so this experiment is a diagnostic stress test, not a general source-classification benchmark. The sweep has only 10 paired seeds per cell. AERCA uses different, strong, possibly multi-source pulses. SMD contains only one selected machine, eight events, and affected-variable labels. No cross-method superiority, real-world RCA accuracy, end-to-end detection performance or new-model contribution is claimed.")
    p("The next comparison should use learnable normal structural dynamics, varied source locations and a common event protocol. For real-data studies, fix constant-channel treatment and dimension indexing independently of evaluation labels. The current package documents implementation progress and bounded preliminary findings.")
    h("Reproducibility and Sources",2)
    p("Case CSVs, uncertainty intervals, arrays, label audits, sensitivity results and checkpoints are retained. Eight automated checks passed. README.md records reproduction commands and provenance; original inputs remain unchanged.")
    p("[1] X. Han, S. Absar, L. Zhang and S. Yuan, Root Cause Analysis of Anomalies in Multivariate Time Series through Granger Causal Discovery, ICLR 2025. https://openreview.net/forum?id=k38Th3x4d9")
    p("[2] Official AERCA source. https://github.com/hanxiao0607/AERCA (exact commit above and in the run manifests).")
    p("[3] Official OmniAnomaly / SMD release. https://github.com/NetManAIOps/OmniAnomaly (commit 7fb0e0acf89ea49908896bcc9f9e80fcfff6baf4).")

    # Both visual formats are generated from the same measured content blocks.
    styles={
        'heading1':ParagraphStyle('h1',fontName='Helvetica-Bold',fontSize=19,leading=23,textColor=colors.HexColor('#173e56'),spaceAfter=13),
        'heading2':ParagraphStyle('h2',fontName='Helvetica-Bold',fontSize=12.5,leading=16,textColor=colors.HexColor('#173e56'),spaceBefore=8,spaceAfter=7),
        'body':ParagraphStyle('body',fontName='Helvetica',fontSize=9.5,leading=13.7,spaceAfter=8),
        'caption':ParagraphStyle('caption',fontName='Helvetica',fontSize=8.2,leading=11,textColor=colors.HexColor('#485763'),spaceAfter=10),
        'cell':ParagraphStyle('cell',fontName='Helvetica',fontSize=8,leading=10.5),
        'headcell':ParagraphStyle('headcell',fontName='Helvetica-Bold',fontSize=8,leading=10.5),
    }
    story=[]; html_blocks=[]; markdown=[]
    for block in blocks:
        kind=block[0]
        if kind=='heading':
            text,level=block[1:]
            story.append(Paragraph(escape(text),styles[f'heading{level}']))
            html_blocks.append(f'<h{level}>{html.escape(text)}</h{level}>'); markdown.append('#'*level+' '+text)
        elif kind=='paragraph':
            text=block[1]
            story.append(Paragraph(escape(text),styles['body']))
            html_blocks.append('<p>'+html.escape(text)+'</p>'); markdown.append(text)
        elif kind=='table':
            _,title,headers,rows,widths=block
            story.append(Paragraph(escape(title),styles['caption']))
            data=[[Paragraph(escape(str(v)),styles['headcell']) for v in headers]]
            data += [[Paragraph(escape(str(v)),styles['cell']) for v in row] for row in rows]
            t=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
            t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e9eff3')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#b4c4ce')),('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#d7e0e5'))]))
            story.extend([t,Spacer(1,10)])
            table_html='<table><caption>'+html.escape(title)+'</caption><thead><tr>'+''.join('<th>'+html.escape(str(v))+'</th>' for v in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table>'
            html_blocks.append(table_html); markdown.append(table_html)
        elif kind=='figure':
            _,name,caption,width=block
            path=OUT/'figures'/name; iw,ih=ImageReader(str(path)).getSize()
            story.extend([Image(str(path),width=width,height=width*ih/iw),Paragraph(escape(caption),styles['caption'])])
            data=base64.b64encode(path.read_bytes()).decode()
            html_blocks.append(f'<figure><img src="data:image/png;base64,{data}" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption></figure>')
            markdown.append(f'![{caption}](figures/{name})\n\n{caption}')
        elif kind=='page':
            story.append(PageBreak()); html_blocks.append('<div class="page-break"></div>')
    def footer(canvas,doc):
        canvas.saveState(); canvas.setStrokeColor(colors.HexColor('#d7e0e5')); canvas.line(72,43,540,43)
        canvas.setFont('Helvetica',8); canvas.setFillColor(colors.HexColor('#52636f'))
        canvas.drawString(72,29,'Anomaly localisation | Preliminary results | 14 September 2026')
        canvas.drawRightString(540,29,str(doc.page)); canvas.restoreState()
    pdf=OUT/'Preliminary_Results.pdf'
    doc=SimpleDocTemplate(str(pdf),pagesize=(612,792),leftMargin=72,rightMargin=72,topMargin=48,bottomMargin=60,title='Preliminary Results and Implementation Progress',author='Experimental analysis for semester-final report')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    style='body{font:16px/1.6 system-ui,sans-serif;color:#26323b;max-width:960px;margin:48px auto;padding:0 24px}h1,h2{color:#173e56;line-height:1.25}h1{font-size:30px}h2{font-size:22px;margin-top:32px}table{width:100%;border-collapse:collapse;font-size:14px;margin:24px 0}th,td{padding:10px;text-align:left;border-bottom:1px solid #d7e0e5}th{background:#e9eff3}caption,figcaption{font-size:13px;color:#485763;text-align:left}caption{margin-bottom:8px}figure{margin:24px 0}img{max-width:100%;height:auto}.page-break{margin-top:36px}@media print{body{margin:0;font-size:10pt}.page-break{break-before:page}h1{font-size:20pt}h2{font-size:14pt}figure,table{break-inside:avoid}}'
    (OUT/'Preliminary_Results.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Preliminary Results</title><style>'+style+'</style><body>'+''.join(html_blocks)+'</body></html>')
    (OUT/'Preliminary_Results.md').write_text('\n\n'.join(markdown)+'\n')
    print(pdf)


if __name__=='__main__':
    main()
