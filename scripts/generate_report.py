#!/usr/bin/env python3
"""Generate one evidence-based report, plots and a clearly marked PDF draft."""
import argparse,json,os,sys
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/pathsnap-mpl')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'frontend'))
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak
from xml.sax.saxutils import escape


def main():
    p=argparse.ArgumentParser();p.add_argument('--comparison',default='outputs/experiments/student_test/model_comparison.csv');a=p.parse_args()
    reports=ROOT/'reports';reports.mkdir(exist_ok=True);figs=ROOT/'outputs/figures';figs.mkdir(parents=True,exist_ok=True)
    comparison=ROOT/a.comparison
    if not comparison.exists():comparison=ROOT/'outputs/runtime/comparison/model_comparison.csv'
    frame=pd.read_csv(comparison) if comparison.exists() else pd.DataFrame()
    charts=[]
    if len(frame):
        fig,ax=plt.subplots(1,2,figsize=(11,4));x=range(len(frame))
        ax[0].bar(x,frame['cv_macro_f1']);ax[0].set_title('Training grouped OOF macro-F1')
        ax[1].bar(x,frame['mean_trip_observation_top1']);ax[1].set_title('Student-test mean trip Top-1')
        for axt in ax:axt.set_xticks(list(x),frame['model'],rotation=25,ha='right');axt.set_ylim(0,1)
        fig.tight_layout();fig.savefig(figs/'model_comparison.png',dpi=150);plt.close(fig);charts.append(figs/'model_comparison.png')
        fig,ax=plt.subplots(figsize=(8,3));ax.bar(frame['model'],frame['candidate_binary_log_loss']);ax.set_title('Student-test candidate binary log loss');fig.tight_layout();fig.savefig(figs/'log_loss.png',dpi=150);plt.close(fig);charts.append(figs/'log_loss.png')
        fig,axes=plt.subplots(1,len(frame),figsize=(14,3))
        import ast
        for ax,(_,row) in zip(axes,frame.iterrows()):
            matrix=ast.literal_eval(row['confusion_matrix']);ax.imshow(matrix,cmap='Blues');ax.set_title(row['model']);ax.set_xticks([0,1]);ax.set_yticks([0,1])
            for i in range(2):
                for j in range(2):ax.text(j,i,str(matrix[i][j]),ha='center',va='center',fontsize=8)
        fig.tight_layout();fig.savefig(figs/'confusion_matrices.png',dpi=150);plt.close(fig);charts.append(figs/'confusion_matrices.png')
    api_file=ROOT/'outputs/verification/api_trip_metrics.csv';api=pd.read_csv(api_file) if api_file.exists() else pd.DataFrame()
    tuning_path=ROOT/'outputs/experiments/tuning_v2/tuning_summary.json'
    tuning=json.loads(tuning_path.read_text()) if tuning_path.exists() else None
    if tuning:
        trials=pd.read_csv(tuning_path.parent/'trials.csv')
        complete=trials[trials['state']=='COMPLETE']
        if len(complete):
            fig,ax=plt.subplots(figsize=(8,3));ax.plot(complete['number'],complete['value'],'o-',label='Trial');ax.plot(complete['number'],complete['value'].cummax(),label='Best so far');ax.set_title('Actual Bayesian tuning history');ax.legend();fig.tight_layout();fig.savefig(figs/'tuning_history.png',dpi=150);plt.close(fig);charts.append(figs/'tuning_history.png')
    history_path=ROOT/'outputs/experiments/models/pytorch_mlp/loss_history.json'
    if history_path.exists():
        history=json.loads(history_path.read_text())
        if history:
            fig,ax=plt.subplots(figsize=(8,3))
            for label,losses in history.items():ax.plot(range(1,len(losses)+1),losses,label=label)
            ax.set_title('Recorded PyTorch training loss; fixed epochs, no validation curve');ax.set_xlabel('Epoch');ax.set_ylabel('BCE');ax.legend();fig.tight_layout();fig.savefig(figs/'mlp_training_loss.png',dpi=150);plt.close(fig);charts.append(figs/'mlp_training_loss.png')
    if len(api):
        fig,ax=plt.subplots(figsize=(10,4));x=list(range(len(api)));ax.plot(x,api['classifier_top1'],'o-',label='Classifier');ax.plot(x,api['decoded_top1'],'o-',label='Classifier + Viterbi');ax.set_xticks(x,api['trip_id'].astype(str).str.zfill(8),rotation=30);ax.set_ylim(0,1);ax.legend();ax.set_title('Actual API predictions: provisional-reference Top-1');fig.tight_layout();fig.savefig(figs/'decoder_comparison.png',dpi=150);plt.close(fig);charts.append(figs/'decoder_comparison.png')
    audit_path=ROOT/'outputs/audit/audit_summary.json';audit=json.loads(audit_path.read_text()) if audit_path.exists() else {}
    sections=[('Status and scope','Development report. Full retraining and the new tuning workflow are to be executed by the user. Saved-model inference and synthetic regression results are recorded separately. Physical Laptop 2 and official scoring remain pending.'),
    ('Dataset understanding and EDA',json.dumps(audit)),
    ('Preprocessing and features','The raw files are preserved. Per-trip WGS84-assumed azimuthal-equidistant projection, original arc/node IDs, long-segment-safe STRtree candidates, and 22 relative geometry/motion/topology features are used. See docs/FEATURE_AUDIT.md. Candidate budgets count directed arcs and ties break by arc ID. Radius expansion occurs only when no candidate is retained. This can miss an offset true road when a closer false road exists; final recall is measured, not presumed.'),
    ('Reference limitations','Labels use ordered route-occurrence dynamic programming, not a row-wise zip or route-membership target. Labels are derived references, not independently established physical truth. A pointwise organiser scoring convention remains unresolved. Reference residuals above 300 m are unknown; coverage and true candidate misses are reported separately.'),
    ('Model experimentation','Logistic regression uses fold-local imputation/scaling and logistic loss. Decision trees use regularized depth/leaf size and Gini impurity; random forests average bounded-depth trees. XGBoost uses binary logistic loss and histogram boosting. The two-hidden-layer PyTorch network uses batched BCE-with-logits and fold-local scaling. Strengths: linear baseline interpretability; tree nonlinear interactions; forest variance reduction; boosting flexible ranking; MLP learned nonlinear interactions. Limitations: linear feature interactions; individual tree variance; forest size/runtime; boosting search/overfit risk; MLP scale/sensitivity. Configurations and genuine loss histories are saved with each new training run. Existing CV and student-test results are development estimates. Training accuracy alone is not evidence of generalization.'),
    ('Bayesian tuning','The repeatable workflow selects a family from grouped training OOF mean-per-trip Top-1; candidate macro-F1, hit-conditional observation loss, trip variation and fit cost break ties. TPE tunes whichever of the five families wins, with 15 total trials and five startup trials. Studies are persistent and keyed to feature/model/search identity. Tuned/untuned student-test comparison and best parameters are emitted by the user-run command. No new full-data tuning gain is claimed in this report.'),
    ('HMM and Viterbi','States are directed arc IDs plus projected fractions. Emission potentials are clipped positive classifier probabilities normalized within each observation, a hybrid discriminative approximation rather than a calibrated generative likelihood. The log-space recurrence maximizes previous score minus absolute directed travel/GPS displacement discrepancy divided by beta, plus log emission. Directed shortest paths use edge-aware adjacency, bounded source caches and original arc reconstruction. Same-arc backwards jitter is bounded by 12 m. Gaps over 60 seconds, missing candidates and impossible transitions create independently backtracked blocks. Intermediate arcs are predictions; no geometry bridges disconnected blocks. Turn restrictions were not supplied.'),
    ('API, UI and repeated pipeline','Laptop 1 serves a checksummed model without retraining. /health, /trips, /predict/{trip_id}, /jobs/{job_id}, /results/{job_id} implement bounded single-process jobs and safe failure states. Private scoring runs only after prediction. Laptop 2 contains Streamlit, an HTTP client and a self-contained Folium/Leaflet vector map. It displays actual decoded/classifier/network/GPS layers, scores, stage/status and preserved results across reruns. A bounded cycle runner joins audit, preparation, five-model fitting, comparison, tuning, export, tests, HTTP verification, report and packaging.'),
    ('Verification and limitations','Synthetic regression tests cover direction, duplicate edges, zero lengths, projection, gaps, missing candidates, decoder restarts, candidate misses, model CPU serialization, API failure/readiness/concurrency, and offline map dependencies. Actual development API verification is in outputs/verification/. A localhost test does not prove physical two-laptop connectivity. Reserved evaluator references were not accessed for development. Newly changed training/tuning code is not represented as a completed full-data experiment.'),
    ('Reproducibility and AI assistance','Run the exact commands in README.md and docs/TWO_LAPTOP_DEMO.md. AI assistance was used for implementation, debugging and report generation. Tables and figures are generated from saved experimental outputs and executed checks; team members must verify and be able to explain the work. Team identities are not inferred.'),
    ('References','Official supplied Main Guidelines, Dataset Exploration, Models and Training Guidelines, Deliverables, Assessment and Plagiarism Policy PDFs and the hackathon DOCX were reviewed. Method references: https://www.microsoft.com/en-us/research/publication/hidden-markov-map-matching-noise-sparseness/ ; https://shapely.readthedocs.io/en/stable/strtree.html ; https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html ; https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html ; https://xgboost.readthedocs.io/en/stable/tutorials/param_tuning.html')]
    text='# PathSnap technical report — development draft\n\n'+'\n\n'.join('## '+h+'\n\n'+t for h,t in sections)
    if tuning:
        sections=[(h,('Executed tuning evidence: '+json.dumps(tuning)) if h=='Bayesian tuning' else t) for h,t in sections]
        text+='\n\nExecuted tuning evidence:\n\n'+json.dumps(tuning,indent=2)
    if len(frame):text+='\n\nActual model comparison:\n\n'+frame.to_csv(index=False)
    if len(api):text+='\n\nExecuted API comparison:\n\n'+api.to_csv(index=False)
    (reports/'PathSnap_Technical_Report.md').write_text(text)
    styles=getSampleStyleSheet();story=[Paragraph('PathSnap technical report',styles['Title']),Paragraph('Development draft · actual saved-model evidence',styles['Heading2'])]
    for heading,body in sections:
        story.extend([Paragraph(escape(heading),styles['Heading2']),Paragraph(escape(body),styles['BodyText']),Spacer(1,8)])
        if heading=='Model experimentation' and len(frame):
            columns=['model','cv_macro_f1','mean_trip_observation_top1','candidate_macro_f1']
            data=[['Model','CV macro-F1','Test trip Top-1','Test macro-F1']]+[[str(r[c]) if c=='model' else f'{r[c]:.4f}' for c in columns] for _,r in frame.iterrows()]
            table=Table(data);table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.lightgrey),('FONTSIZE',(0,0),(-1,-1),8),('GRID',(0,0),(-1,-1),.3,colors.grey)]));story.append(table)
    for image in charts:
        story.extend([PageBreak(),Paragraph(image.stem.replace('_',' '),styles['Heading2']),Image(str(image),width=6.6*inch,height=2.6*inch)])
    for name in ['ui.png','map.png','api_log.png']:
        image=ROOT/'outputs/verification'/name
        if image.exists():story.extend([PageBreak(),Paragraph('Genuine local screenshot: '+name,styles['Heading2']),Image(str(image),width=6.5*inch,height=4*inch)])
    SimpleDocTemplate(str(reports/'PathSnap_Technical_Report.pdf')).build(story)
    print(reports/'PathSnap_Technical_Report.pdf')
if __name__=='__main__':main()
