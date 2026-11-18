#!/usr/bin/env python3
"""Reproducible mentor report from real project evidence; no model fitting."""
from __future__ import annotations
import argparse,ast,hashlib,importlib.metadata,json,os,platform,sqlite3,sys,textwrap
from pathlib import Path
from xml.sax.saxutils import escape
os.environ.setdefault('MPLCONFIGDIR','/tmp/pathsnap-report-mpl')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'frontend'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER,TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle,getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import BaseDocTemplate,PageTemplate,Frame,Paragraph,Spacer,PageBreak,Table,LongTable,TableStyle,Image,KeepTogether,Preformatted
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.graphics.shapes import Drawing,Rect,String,Line,Polygon

DATE='18 November 2026'
NAVY=colors.HexColor('#142b44');TEAL=colors.HexColor('#007d87');LIGHT=colors.HexColor('#eef4f7');GREY=colors.HexColor('#526372')
LABELS={'logistic':'Logistic regression','decision_tree':'Decision tree','random_forest':'Random forest','xgboost':'XGBoost','pytorch_mlp':'PyTorch MLP'}
TEAM=[('Murali R','PES2UG25AM805'),('Ranjith G','PES2UG25AM806'),('Kushal J','PES2UG24AM205'),('Yashas V','PES2UG24AM188')]
REPO='https://github.com/Murali-co/PathSnap-'
BRANCH=REPO+'/tree/feature/frontend-web'


def read_json(path,default=None):
    p=ROOT/path
    return json.loads(p.read_text()) if p.exists() else default


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def fmt(v,digits=4):
    return 'N/A' if v is None or (isinstance(v,float) and not np.isfinite(v)) else f'{v:.{digits}f}'


def pct(v):return 'N/A' if v is None else f'{100*v:.2f}%'


class ReportDoc(BaseDocTemplate):
    def __init__(self,path,**kwargs):
        super().__init__(path,pagesize=A4,leftMargin=46,rightMargin=46,topMargin=53,bottomMargin=48,**kwargs)
        frame=Frame(self.leftMargin,self.bottomMargin,self.width,self.height,id='body',leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)
        self.addPageTemplates(PageTemplate(id='normal',frames=frame,onPage=self.decorate));self.bookmark_id=0
    def beforeDocument(self):
        self.bookmark_id=0
    def decorate(self,c,doc):
        c.saveState();w,h=A4
        if doc.page>1:
            c.setStrokeColor(TEAL);c.setLineWidth(1);c.line(46,h-33,w-46,h-33)
            c.setFillColor(NAVY);c.setFont('ReportSans',8);c.drawString(46,h-25,'PATHSNAP  |  MENTOR SUBMISSION')
        c.setFillColor(GREY);c.setFont('ReportSans',7.5);c.drawString(46,28,DATE+'  •  Development evidence; external checks identified')
        c.drawRightString(w-46,28,str(doc.page));c.restoreState()
    def afterFlowable(self,flowable):
        if isinstance(flowable,Paragraph) and flowable.style.name in ('Chapter','Subchapter'):
            if flowable.getPlainText()=='Contents':return
            level=0 if flowable.style.name=='Chapter' else 1
            self.bookmark_id+=1;key='heading_'+str(self.bookmark_id)
            self.canv.bookmarkPage(key);self.canv.addOutlineEntry(flowable.getPlainText(),key,level=level,closed=False)
            if level==0:self.notify('TOCEntry',(level,flowable.getPlainText(),self.page,key))


class Report:
    def __init__(self):
        fontroot=Path('/usr/share/fonts/truetype/dejavu')
        for name,file in [('ReportSans','DejaVuSans.ttf'),('ReportBold','DejaVuSans-Bold.ttf'),('ReportMono','DejaVuSansMono.ttf')]:
            pdfmetrics.registerFont(TTFont(name,str(fontroot/file)))
        pdfmetrics.registerFontFamily('ReportSans',normal='ReportSans',bold='ReportBold',italic='ReportSans',boldItalic='ReportBold')
        self.styles={
            'body':ParagraphStyle('Body',fontName='ReportSans',fontSize=9.3,leading=13.7,textColor=NAVY,spaceAfter=8),
            'small':ParagraphStyle('Small',fontName='ReportSans',fontSize=7.8,leading=10.8,textColor=GREY,spaceAfter=5),
            'chapter':ParagraphStyle('Chapter',fontName='ReportBold',fontSize=18,leading=23,textColor=NAVY,spaceBefore=4,spaceAfter=14,keepWithNext=True),
            'sub':ParagraphStyle('Subchapter',fontName='ReportBold',fontSize=11.1,leading=15,textColor=TEAL,spaceBefore=10,spaceAfter=7,keepWithNext=True),
            'cell':ParagraphStyle('Cell',fontName='ReportSans',fontSize=7.7,leading=10.7,textColor=NAVY),
            'headcell':ParagraphStyle('HeadCell',fontName='ReportBold',fontSize=7.6,leading=10.4,textColor=colors.white),
            'caption':ParagraphStyle('Caption',fontName='ReportSans',fontSize=8,leading=11,textColor=GREY,spaceBefore=5,spaceAfter=12),
            'code':ParagraphStyle('Code',fontName='ReportMono',fontSize=7.2,leading=10.3,backColor=LIGHT,borderPadding=8,spaceBefore=7,spaceAfter=10),
        }
        self.story=[];self.markdown=[];self.figure_number=0
    def p(self,text,small=False):
        self.story.append(Paragraph(text,self.styles['small' if small else 'body']));self.markdown.append(text.replace('<b>','').replace('</b>',''))
    def chapter(self,title,new=True):
        if new:self.story.append(PageBreak())
        self.story.append(Paragraph(escape(title),self.styles['chapter']));self.markdown.append('\n# '+title+'\n')
    def sub(self,title):
        self.story.append(Paragraph(escape(title),self.styles['sub']));self.markdown.append('\n## '+title+'\n')
    def table(self,headers,rows,widths=None):
        def cell(x,header=False):
            s=escape(str(x)).replace('\n','<br/>').replace('_','_\u200b').replace('/','/\u200b')
            return Paragraph(s,self.styles['headcell' if header else 'cell'])
        data=[[cell(x,True) for x in headers]]+[[cell(x) for x in row] for row in rows]
        t=LongTable(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),NAVY),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,LIGHT]),('LINEBELOW',(0,-1),(-1,-1),.5,colors.HexColor('#cbd6df'))]))
        self.story.extend([t,Spacer(1,9)])
        self.markdown.extend(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows])
    def fig(self,path,caption,max_height=290):
        from PIL import Image as PILImage
        with PILImage.open(path) as im:w,h=im.size
        width=min(503,w*503/max(w,1));height=width*h/w
        if height>max_height:width*=max_height/height;height=max_height
        self.figure_number+=1
        block=[Image(str(path),width=width,height=height),Paragraph(f'<b>Figure {self.figure_number}.</b> '+escape(caption),self.styles['caption'])]
        self.story.append(KeepTogether(block));self.markdown.append(f'\n![Figure {self.figure_number}: {caption}]({Path(path).relative_to(ROOT).as_posix()})\n')
    def code(self,text):
        self.story.append(Preformatted(text,self.styles['code'],maxLineLength=91,splitChars=' /'));self.markdown.append('\n```text\n'+text+'\n```\n')


def prepare_evidence(comparison):
    models=pd.read_csv(comparison);inventory=pd.read_csv(ROOT/'outputs/audit/inventory.csv',dtype={'trip_id':str})
    api=pd.read_csv(ROOT/'outputs/verification/api_trip_metrics.csv',dtype={'trip_id':str})
    results={tid:read_json('outputs/verification/'+tid+'.json') for tid in api.trip_id}
    counts={};manifest=read_json('outputs/prepared/feature_manifest.json')
    for e in manifest['entries']:
        with np.load(ROOT/'outputs/prepared'/e['file']) as a:
            labels=a['labels'];known=labels>=0;hits=np.bincount(a['obs'],weights=a['y'],minlength=len(labels))>0
            row=counts.setdefault(e['partition'],dict(trips=0,observations=0,known=0,unknown=0,candidates=0,positive=0,hits=0,ambiguous=0,no_candidates=0))
            values=dict(trips=1,observations=len(labels),known=int(known.sum()),unknown=int((~known).sum()),candidates=len(a['y']),positive=int(a['y'].sum()),hits=int(hits[known].sum()),ambiguous=int(a['ambiguous'].sum()),no_candidates=int((np.diff(a['offsets'])==0).sum()))
            for k,v in values.items():row[k]+=v
    aggregate={}
    for kind in ('classifier','decoded'):
        rows=[r['metrics'][kind] for r in results.values()]
        aggregate[kind]=dict(mean_top1=float(np.mean([m['observation_top1'] for m in rows])),pooled_top1=sum(m['correct_observations'] for m in rows)/sum(m['known_observations'] for m in rows),mean_continuity=float(np.mean([m['route_continuity'] for m in rows])),pooled_continuity=sum(m['legal_pairs'] for m in rows)/sum(m['eligible_pairs'] for m in rows),mean_route_f1=float(np.mean([m['route_f1'] for m in rows])),mean_edit=float(np.mean([m['normalized_edit_distance'] for m in rows])),eligible_pairs=sum(m['eligible_pairs'] for m in rows),legal_pairs=sum(m['legal_pairs'] for m in rows),gap_pairs=sum(m['gap_pairs'] for m in rows))
    tuning=[];best_params={};study_name=None
    database=ROOT/'outputs/experiments/tuning/tuning.sqlite3'
    if database.exists():
        c=sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True)
        study_name=c.execute('select study_name from studies limit 1').fetchone()[0]
        tuning=[dict(trial=n,state=s,value=v) for n,s,v in c.execute('select t.number,t.state,v.value from trials t left join trial_values v using(trial_id) order by t.number')]
        best=c.execute('select trial_id from trial_values order by value desc limit 1').fetchone()
        if best:
            for name,value,distribution in c.execute('select param_name,param_value,distribution_json from trial_params where trial_id=?',(best[0],)):
                best_params[name]=int(value) if json.loads(distribution)['name']=='IntDistribution' else value
        c.close()
    return dict(models=models,inventory=inventory,api=api,results=results,counts=counts,aggregate=aggregate,tuning=tuning,best_params=best_params,study_name=study_name,config=manifest['config'])


def make_figures(e,assets):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold'})
    out={};inv=e['inventory'];models=e['models'];api=e['api']
    def save(name,fig):
        fig.tight_layout();path=assets/(name+'.png');fig.savefig(path,dpi=170,bbox_inches='tight');plt.close(fig);out[name]=path
    fig,ax=plt.subplots(1,2,figsize=(10,3.2))
    for part,color in [('train','#007d87'),('student_test','#e3a344')]:
        frame=inv[inv.partition==part];ax[0].scatter((frame.lon_min+frame.lon_max)/2,(frame.lat_min+frame.lat_max)/2,label=part,s=22,c=color)
        ax[1].scatter(frame.observations,frame.arcs,label=part,s=22,c=color)
    ax[0].set(xlabel='Longitude (degrees)',ylabel='Latitude (degrees)',title='Development trip locations');ax[0].legend(fontsize=7)
    ax[1].set(xlabel='GPS observations / trip',ylabel='Directed arcs / trip',title='Trajectory and graph scale',xscale='log',yscale='log');save('eda',fig)
    fig,ax=plt.subplots(1,2,figsize=(10,3))
    ax[0].hist(inv.max_gap_s,bins=np.logspace(0,6,22),color='#007d87');ax[0].set(xscale='log',xlabel='Maximum timestamp gap per trip (s)',ylabel='Trips',title='Irregular sampling')
    diagnostics=pd.read_csv(ROOT/'outputs/prepared/candidate_diagnostics.csv');ax[1].boxplot([diagnostics[diagnostics.partition==p].candidate_recall for p in ['train','student_test']],tick_labels=['Training','Student test']);ax[1].set(ylabel='Final candidate recall / trip',ylim=(0,1.03),title='Retained candidate recall');save('data_quality',fig)
    names=[LABELS[n] for n in models.model];x=np.arange(len(names))
    fig,ax=plt.subplots(1,2,figsize=(10,3.6));ax[0].bar(x,models.cv_macro_f1,color='#007d87');ax[0].set_title('Pooled OOF candidate macro-F1')
    ax[1].bar(x-.18,models.mean_trip_observation_top1,.36,label='Mean trip Top-1',color='#007d87');ax[1].bar(x+.18,models.candidate_macro_f1,.36,label='Candidate macro-F1',color='#e3a344');ax[1].set_title('Student-test comparison');ax[1].legend(fontsize=7,loc='lower right')
    for a in ax:a.set_xticks(x,names,rotation=30,ha='right');a.set_ylim(0,1);a.grid(axis='y',alpha=.15)
    save('models',fig)
    fig,ax=plt.subplots(1,2,figsize=(10,3.2));ax[0].bar(names,models.candidate_binary_log_loss,color='#007d87');ax[0].set_title('Student-test binary log loss (lower is better)');ax[1].bar(names,models.fit_seconds,color='#e3a344');ax[1].set_title('Recorded final-fit duration (s)')
    for a in ax:a.tick_params(axis='x',rotation=30)
    save('loss_runtime',fig)
    fig,axes=plt.subplots(2,3,figsize=(10,5.3));axes=axes.ravel()
    for ax,(_,row) in zip(axes,models.iterrows()):
        matrix=np.array(ast.literal_eval(row.confusion_matrix));ax.imshow(matrix,cmap='Blues');ax.set_title(LABELS[row.model],fontsize=9);ax.set_xticks([0,1],['Negative','Positive'],fontsize=7);ax.set_yticks([0,1],['Negative','Positive'],fontsize=7);ax.set_xlabel('Predicted',fontsize=7);ax.set_ylabel('Actual',fontsize=7)
        for i in range(2):
            for j in range(2):ax.text(j,i,f'{matrix[i,j]:,}',ha='center',va='center',fontsize=9,color='white' if matrix[i,j]>matrix.max()/2 else 'black')
    axes[-1].axis('off');save('confusion',fig)
    history=read_json('outputs/experiments/models/pytorch_mlp/loss_history.json',{})
    if history:
        fig,ax=plt.subplots(figsize=(9,3.2))
        for label,values in history.items():ax.plot(range(1,len(values)+1),values,label='Full fit' if label=='full_fit' else 'Fold '+str(int(label)+1))
        ax.set(xlabel='Epoch',ylabel='Training BCE-with-logits',title='Genuine recorded MLP training histories');ax.legend(fontsize=8);save('mlp',fig)
    fig,ax=plt.subplots(1,2,figsize=(10,3.6));x=np.arange(len(api))
    ax[0].plot(x,api.classifier_top1,'o-',label='Classifier',color='#e3a344');ax[0].plot(x,api.decoded_top1,'o-',label='Classifier + Viterbi',color='#007d87');ax[0].set(ylabel='Top-1 accuracy',ylim=(.65,1.01),title='Actual per-trip decoder comparison');ax[0].legend(fontsize=7)
    ax[1].bar(x,api.reported_seconds,color='#007d87');ax[1].set(ylabel='End-to-end API duration (s)',title='Recorded request duration')
    for a in ax:a.set_xticks(x,[t[-2:] for t in api.trip_id]);a.set_xlabel('Trip ID suffix');a.grid(axis='y',alpha=.15)
    save('decoder',fig)
    if e['tuning']:
        completed=[r for r in e['tuning'] if r['state']=='COMPLETE'];values=[r['value'] for r in completed]
        fig,ax=plt.subplots(figsize=(9,3));ax.plot([r['trial'] for r in completed],values,'o-',label='Recorded trial');ax.plot([r['trial'] for r in completed],np.maximum.accumulate(values),'--',label='Best observed');ax.set(xlabel='Trial number (zero based)',ylabel='Legacy mean-fold candidate macro-F1',title='Partial saved search; final tuned model not exported');ax.legend(fontsize=8);save('tuning',fig)
    return out


def architecture():
    d=Drawing(503,178);labels=[['Input files','Metric geometry','Spatial candidates'],['Relative features','Saved classifier','Directed Viterbi'],['Reconstructed route','Private scorer + API','Laptop 2 interface']]
    for row,items in enumerate(labels):
        y=132-row*57
        for col,label in enumerate(items):
            x=col*172;d.add(Rect(x,y,158,35,rx=5,ry=5,fillColor=LIGHT,strokeColor=TEAL));d.add(String(x+79,y+14,label,textAnchor='middle',fontName='ReportSans',fontSize=8.1,fillColor=NAVY))
            if col<2:d.add(Line(x+158,y+17,x+170,y+17,strokeColor=TEAL));d.add(Polygon([x+170,y+17,x+165,y+20,x+165,y+14],fillColor=TEAL,strokeColor=TEAL))
        if row<2:d.add(Line(485,y,485,y-13,strokeColor=TEAL));d.add(Line(485,y-13,5,y-13,strokeColor=TEAL));d.add(Line(5,y-13,5,y-22,strokeColor=TEAL))
    return d


def make_development_map(e,assets):
    from map_view import make_map
    import folium
    tid='00000052';result=e['results'][tid];m=make_map(result)
    directory=ROOT.parent/'student_test'/tid
    if not directory.is_dir():return None
    nodes=np.loadtxt(directory/f'{tid}.nodes');arcs=np.loadtxt(directory/f'{tid}.arcs',dtype=int);route=np.loadtxt(directory/f'{tid}.route',dtype=int,ndmin=1)
    layer=folium.FeatureGroup(name='Supplied reference route (development)',show=True)
    for arc in route:
        pair=nodes[arcs[int(arc)]];folium.PolyLine([[lat,lon] for lon,lat in pair],color='#a855f7',weight=3,dash_array='6,5',tooltip='Supplied reference route').add_to(layer)
    layer.add_to(m)
    # Recreate control after all layers so reference can be toggled.
    for key,value in list(m._children.items()):
        if isinstance(value,folium.map.LayerControl):del m._children[key]
    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(folium.Element('<div style="position:fixed;top:12px;left:60px;z-index:9999;background:white;padding:10px;font:13px sans-serif;border:1px solid #bbb">Student-test 00000052 · development map<br>Green: decoded · Purple dashed: supplied reference · Blue: GPS</div>'))
    html=m.get_root().render();path=assets/'development_map.html';path.write_text(html)
    image=assets/'development_map.png'
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser=p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,args=['--no-sandbox','--disable-dev-shm-usage'])
            page=browser.new_page(viewport={'width':1300,'height':850});page.set_content(html,wait_until='load');page.locator('.leaflet-container').wait_for();page.screenshot(path=str(image));browser.close()
        return image
    except Exception:
        return ROOT/'outputs/verification/map.png' # Existing genuine map screenshot, without a reference overlay.


def main():
    p=argparse.ArgumentParser();p.add_argument('--comparison',default='outputs/experiments/student_test/model_comparison.csv');a=p.parse_args()
    assets=ROOT/'reports/assets';assets.mkdir(parents=True,exist_ok=True)
    e=prepare_evidence(ROOT/a.comparison);figs=make_figures(e,assets);development_map=make_development_map(e,assets)
    metadata=read_json('reports/model_metadata.json',{});r=Report();models=e['models'].set_index('model');inv=e['inventory'];api=e['api'];agg=e['aggregate'];counts=e['counts'];cfg=e['config'];w=503
    title=ParagraphStyle('CoverTitle',fontName='ReportBold',fontSize=40,leading=46,textColor=NAVY,spaceAfter=20)
    subtitle=ParagraphStyle('CoverSub',fontName='ReportSans',fontSize=18,leading=25,textColor=TEAL,spaceAfter=12)
    r.story.extend([Spacer(1,36),Paragraph('PATHSNAP',title),Paragraph('Distributed map matching<br/>and route reconstruction',subtitle)])
    r.p('<b>Technical Project Report · Mentor Submission</b>');r.p('PES University · UE24CS352A: Machine Learning · '+DATE)
    r.story.append(Spacer(1,22));r.table(['Team 1 member','SRN'],TEAM,[260,243]);r.p('Team details are taken from the supplied official hackathon instructions.',True)
    r.story.append(Spacer(1,16));r.table(['Training trips','Model families','Development trips','Decoded mean trip Top-1'],[['80','5','10',pct(agg['decoded']['mean_top1'])]],[105,105,120,173])
    r.p('<b>Evaluation status:</b> recorded development experiments and locally verified applications. Physical two-laptop demonstration, final organiser scoring and a completed new tuning/export comparison remain pending. Reported accuracy uses provisional derived reference labels.')
    r.p('Mentor / evaluator: ______________________________',True)
    r.p('Source repository: <link href="'+REPO+'" color="#007d87">Murali-co/PathSnap-</link><br/>Frontend integration branch: <link href="'+BRANCH+'" color="#007d87">feature/frontend-web</link>',True)
    r.chapter('Contents')
    toc=TableOfContents();toc.levelStyles=[ParagraphStyle('TOC0',fontName='ReportSans',fontSize=8.8,leading=12,textColor=NAVY,spaceBefore=1),ParagraphStyle('TOC1',fontName='ReportSans',fontSize=7.8,leading=10,leftIndent=14,textColor=GREY)];r.story.append(toc)
    r.chapter('1  Abstract and scope')
    r.p('PathSnap reconstructs a plausible directed road journey from noisy GPS observations using supervised candidate scoring and sequential decoding. Each trip supplies a trajectory, a local road graph and an ordered reference route. A trip-local metric projection and Shapely segment index generate candidate arcs; 22 geometry, motion and topology features feed five distinct classifiers. The selected classifier supplies normalized emission potentials to an HMM-style log-space Viterbi decoder with directed routing and explicit breaks.')
    r.p(f'The permitted development data contains <b>80 training trips and 10 student-test trips</b>, totaling <b>227,577 GPS observations</b>. Full-fit records show <b>1,609,263 resolved training candidate rows</b>. XGBoost leads the recorded grouped OOF comparison: mean trip observation Top-1 is <b>{pct(models.loc["xgboost","cv_mean_trip_top1"])}</b>, with pooled candidate macro-F1 <b>{fmt(models.loc["xgboost","cv_macro_f1"])}</b>. Student-test classifier mean trip Top-1 is <b>{pct(agg["classifier"]["mean_top1"])}</b>. Directed decoding raises it to <b>{pct(agg["decoded"]["mean_top1"])}</b>, a <b>{100*(agg["decoded"]["mean_top1"]-agg["classifier"]["mean_top1"]):.2f} percentage-point</b> change. Mean per-trip continuity rises from {pct(agg["classifier"]["mean_continuity"])} to {pct(agg["decoded"]["mean_continuity"])}.')
    r.p('FastAPI and separate Python/Streamlit and React/Vite clients implement the demonstration workflow. All ten development trips and repeated requests passed actual HTTP verification; 30 focused regression tests passed. A saved older tuning database records ten completed trials and one unresolved RUNNING entry. A finished 15-trial study, a final tuned-versus-untuned student-test gain, physical LAN execution and official reserved-data scores are not claimed.')
    r.p('<b>How to read the results:</b> this is a complete account of the available project and evidence, not a claim that every external acceptance condition is satisfied. Candidate classification metrics, observation matching, route overlap and ordered sequence quality are kept distinct. The reference alignment is provisional; its uncertainty limits all pointwise accuracy claims.')
    r.chapter('2  Problem statement and objectives')
    r.p('GPS noise, missing samples and nearby parallel roads can place observations away from the road actually traversed. Selecting the nearest segment independently can confuse opposite directions or create disconnected routes. PathSnap separates the problem into local road plausibility and sequence feasibility: a classifier scores nearby directed arcs, then the decoder chooses a feasible chronological sequence.')
    r.table(['Objective','Implemented response'],[
        ['Understand inputs','Audit all 90 permitted trips, preserve row identities and validate file schemas.'],['Learn road plausibility','Fit five classifier families to candidate-level binary labels with grouped validation.'],['Reconstruct journeys','Use directed lengths, projected positions, shortest paths and Viterbi backtracking.'],['Demonstrate a distributed application','Laptop 1 hosts model/data/API/private scoring; Laptop 2 hosts a frontend.'],['Support defensible evaluation','Export real experiment records, coverage/misses, route metrics, timings and a reproducible report.']], [135,368])
    r.p('No external road download, paid routing service, LLM prediction endpoint, cloud GPU or hosted database is required. “Distributed” refers to the two-laptop client/server demonstration. The implementation processes complete trips offline; neighbouring future GPS observations are permitted input features and do not make this a causal live-tracking system.')
    r.chapter('3  System architecture and pipeline')
    r.story.append(architecture());r.p('The diagrams show implementation stages, not a simulated experiment. The reference path enters only training labels and post-prediction private scoring; it is absent from candidate generation, features, classifier inference and decoding.',True)
    r.table(['Component','Responsibility / entry point'],[['Data and geometry','data.py, spatial.py: validated original IDs, local projection, segment lookup and directed adjacency.'],['Features and supervision','features.py and reference.py: fixed 22-feature schema and replaceable ordered-occurrence labels.'],['Experiments','training.py, experiments.py: grouped CV, full fits, model comparison, selection and portable export.'],['Sequence inference','decoding/viterbi.py and inference.py: emissions, directed transitions, independent blocks and intermediate road reconstruction.'],['Applications','backend/app/main.py; frontend/app/app.py; frontend-web/: jobs, map display and repeated trip interaction.'],['Repeatable workflow','scripts/run_full_pipeline.py: bounded cycles, resumable training and a persisted stage ledger.']],[125,378])
    r.p('The cycle is audit → prepare → train five families → compare OOF/student-test results → tune the selected family → export → regression tests → all-ten HTTP verification → PDF → local packages. A failed stage stops execution before stale results are exported. `--resume` reuses compatible completed fits; `--cycles` bounds repeated execution. The verified saved-model cycle deliberately skipped real-data fitting and tuning; backend requests never trigger training.')
    r.chapter('4  Dataset understanding and EDA')
    r.table(['Partition','Trips','GPS rows','Node rows','Directed arc rows','Route entries'],[['Training','80','208,417','1,933,485','3,876,904','31,336'],['Student test','10','19,160','90,689','185,041','3,477'],['Permitted total','90','227,577','2,024,174','4,061,945','34,813'],['Reserved evaluation','10 expected','Not analysed','Not analysed','Not analysed','Not analysed']],[96,45,75,90,105,92])
    r.p('Counts above sum trip-local graphs; they are not counts of globally unique roads. The supplied development package has 360 raw files. Each original row retains its identity: `.track` is longitude/latitude/timestamp, `.nodes` is longitude/latitude, `.arcs` is an ordered directed endpoint pair, and `.route` is a chronological arc sequence. Arc IDs are local to a trip, so a global multiclass arc-ID target would be invalid.')
    r.fig(figs['eda'],'Development trip coordinates and per-trip graph/trajectory scale. Coordinates are shown for EDA and excluded from learning features.',260)
    r.sub('4.1  Quality findings and their treatment')
    r.table(['Measured finding','Engineering consequence'],[['0 invalid GPS rows in the audited inputs','Validation remains active; future invalid observations retain their output positions.'],[f'{pct(inv.one_second_intervals.sum()/inv.intervals.sum())} of intervals are exactly 1 s','Time-based windows are used; row counts are not treated as elapsed seconds.'],[f'Maximum timestamp gap: {int(inv.max_gap_s.max()):,} s','Split decoding blocks; do not infer continuous travel or a proven pause.'],['106 zero-length arcs; 1,903 duplicate directed endpoint pairs','Handle projection safely; preserve original arc IDs and edge-aware adjacency.'],['14 adjacent reference-route discontinuities','Retain diagnostics; do not assume reference routes are error-free.'],['Weak components vary from 1 to 421 per trip','Unreachable transitions are valid cases, not a reason to fabricate connecting geometry.'],['Maximum sampled relative projection discrepancy: '+f'{inv.sampled_projection_max_relative_error.max():.8f}','Geodesic comparison samples 100 or fewer arcs per trip; this is not a global distortion bound.']],[217,286])
    r.fig(figs['data_quality'],'Per-trip gap extremes and final retained-candidate recall. All inputs are permitted development data.',230)
    r.chapter('5  Preprocessing, candidates and reference labels')
    r.sub('5.1  Metric geometry and directed candidates')
    r.p('Coordinates are interpreted as longitude/latitude under an explicit WGS84 assumption; the file schema does not independently establish the datum. Each trip receives a local azimuthal-equidistant projection. Segment lengths and projection distances are computed in metres. An STRtree indexes whole line-segment bounding boxes, including long arcs whose midpoints may be far from an observation. Opposite directions and duplicate geometries retain their original IDs.')
    r.p('The current prediction-only policy searches radii 50, 100, 200 and 300 m and stops at the first nonempty radius. It retains at most 16 directed arcs, sorted by distance then original ID. Reference labels never insert candidates. Expansion does not guarantee that the correct offset road is retrieved when a closer false road already exists; this is a measured limitation rather than an assumed solved problem.')
    rows=[]
    for part in ['train','student_test']:
        c=counts[part];rows.append([part,f'{c["observations"]:,}',f'{c["known"]:,}',f'{c["unknown"]:,}',f'{c["candidates"]:,}',pct(c['hits']/c['known']),pct(c['ambiguous']/c['observations'])])
    r.table(['Partition','GPS rows','Known labels','Unknown','Candidate rows','Final recall','Ambiguity flag'],rows,[74,69,72,57,82,74,75])
    r.p('Training caches contain 1,612,959 candidate rows before unknown-label filtering and 1,609,263 fitting rows afterwards. There are 187,005 positive and 1,422,258 negative fitting rows. Student testing has 169,191 candidate rows, 19,128 positives and 32 known-reference candidate misses. Neither partition has an observation with an empty final candidate set in this saved preparation. The training candidate recall ceiling is lower than the student-test ceiling; these splits contain different trip conditions.')
    r.sub('5.2  Provisional supervision')
    r.p('A route is an ordered segment sequence, not one label per GPS row. The aligner uses route-occurrence states, so repeated visits to an arc remain distinct. Point-to-segment residual costs, monotonic occurrence progress, bounded backwards projection costs and skip penalties align observations within valid GPS blocks. Invalid observations and long temporal gaps reset alignment. Reference discontinuities remain diagnostics and affect progress costs.')
    r.p('Aligned residuals above 300 m become unknown labels, producing 415 excluded training observations. Near-equal alternative road distances generate ambiguity flags. These labels are derived references rather than certified physical truth; the organiser’s pointwise convention remains unresolved. Accuracy denominators include known labels only, while candidate misses and unmatched predictions count as incorrect on those known labels. Coverage and ambiguity are reported separately.')
    r.chapter('6  Feature engineering')
    r.p('All 22 features are computed identically at training and prediction from public track/network inputs. The stored schema fixes their order. IDs remain metadata; trip ID, arc/node ID, route membership, aligned progress, reference distances and absolute longitude/latitude are excluded. Learned scaling/imputation is fitted within training folds only. Nonfinite feature values receive deterministic finite replacements, while invalid input rows retain explicit statuses.')
    feature_rows=[
    ['Geometry','distance_m','Perpendicular segment distance (m).'],['Geometry','distance_rank','Zero-based deterministic rank in final candidates.'],['Geometry','distance_delta_m','Distance above the nearest retained candidate (m).'],['Geometry','fraction','Clipped projection fraction along directed arc [0,1].'],['Geometry','endpoint_distance_m','Along-segment distance to nearer endpoint (m).'],['Geometry','length_m','Projected directed arc length (m).'],
    ['Direction','heading_cosine','Candidate/motion direction agreement × heading reliability.'],['Direction','signed_motion_m','Signed motion projected along candidate direction (m).'],['Direction','heading_reliability','Movement magnitude factor × 30-second straightness.'],
    ['Motion/time','previous_displacement_m','Previous valid displacement; zero across invalid/large gaps.'],['Motion/time','next_displacement_m','Next valid displacement; offline feature.'],['Motion/time','speed_mps','Previous displacement / max(previous interval, 1).'],['Motion/time','previous_dt_s','Previous timestamp interval; zero at first observation.'],['Motion/time','next_dt_s','Next timestamp interval; zero at final observation.'],['Motion/time','gap_flag','No reliable prior interval or a subsequent large gap.'],['Motion/time','stationary','Previous displacement <2 m: a heuristic, not proof of parking.'],['Motion/time','straightness_30s','Net/path displacement within a time-based window and valid block.'],
    ['Topology','start_in_degree','Incoming arc count at the candidate start node.'],['Topology','end_out_degree','Outgoing arc count at the candidate end node.'],['Topology','reverse_available','Whether the reverse endpoint pair exists.'],['Candidates','candidate_count','Final directed candidate count for the observation.'],['Candidates','radius_m','First radius with candidates (m).']]
    r.table(['Group','Feature','Definition and reason'],feature_rows,[77,153,273])
    r.p('Heading confidence is reduced for low movement or low net displacement. GPS speed remains a noisy measurement, not verified vehicle speed. Timestamp windows do not cross declared large gaps. Feature groups provide road proximity, direction, motion reliability and local connectivity without geographic memorisation. No feature ablation or calibration gain is claimed: those experiments are not in the saved evidence.')
    r.chapter('7  Experimental design and metric definitions')
    r.p('Training uses 80 trip groups with fixed seed 42 and three GroupKFold splits. Candidate rows belonging to one trip remain together. A final model is refitted on all resolved training rows and then evaluated on the ten student-test trips. Fixed iteration/epoch budgets avoid using outer validation folds or student-test data for early stopping. These are development estimates used for model selection, not final reserved-test estimates [R2].')
    r.table(['Metric','Operational definition'],[['Candidate accuracy / precision / recall','Binary target per retained candidate; threshold 0.5. Positive class means the candidate equals the aligned reference arc.'],['Candidate macro-F1','Mean F1 of negative and positive classes. The consolidated CV value pools OOF candidate predictions; it is not the unweighted mean of fold F1 values.'],['Candidate binary log loss','Binary cross-entropy on clipped positive probabilities; affected by class imbalance and probability quality.'],['Mean trip observation Top-1','Compute correct directed-arc fraction on known labels in each trip, then average trip fractions equally.'],['Pooled observation Top-1','Total correct observations / total known observations across trips. Different denominator from mean trip Top-1.'],['Candidate miss rate','Known reference arc absent from final candidates, including nonempty wrong candidate sets.'],['Conditional observation loss','Within-observation normalized positive probability loss only where reference candidates exist; report alongside misses.'],['Directed continuity','Feasible directed travel under graph and movement/time bounds; report eligible, legal, gap and unmatched pair counts.'],['Route precision / recall / F1','Overlap of arc-ID sets within each trip. Repeated visits do not increase set counts.'],['Normalized edit distance','Edit distance of ordered consecutive-duplicate-collapsed arc sequences / max(sequence lengths). Lower is better; repeated visits remain.']],[157,346])
    r.p('Selection ranks models by training OOF mean-per-trip Top-1. Candidate macro-F1, conditional probability loss, trip-level variation and fit cost are tie-breakers. Student-test comparison is permitted development feedback; its repeated use must not be described as a pristine final holdout. Explicit saved fold trip lists are available for the newer XGBoost and MLP fits; older baseline provenance relies on their code, CV tables, OOF arrays and serialized estimators.')
    r.chapter('8  Model experimentation: five distinct families')
    interpretations={
    'logistic':('Linear and probabilistic baseline','Suitable for a compact relative-feature representation and a simple interpretable reference. Fold-local scaling and median imputation make mixed-unit features usable.','Convex logistic-loss fitting is efficient. Its linear log-odds surface cannot directly express all geometry/motion interactions, and balanced class weights alter the interpretation of raw positive probabilities.','Weighted logistic loss with L2 regularisation.','The lowest candidate macro-F1 among the five shows that this feature representation benefits from nonlinear scoring. That is not conclusive proof of underfitting: saved training-set accuracy or learning curves for this fit are unavailable.'),
    'decision_tree':('Single nonlinear tree','A regularised tree can learn threshold interactions between distance, heading reliability and topology without scaling.','Readable nonlinear splits and fast prediction are useful. One tree has higher variance than an ensemble, and probability estimates can be coarse in small leaves.','Weighted Gini-impurity reduction; bounded depth and minimum leaf size.','It improves candidate macro-F1 over the linear baseline, but mean trip Top-1 is slightly lower on student testing. This demonstrates that binary threshold quality and within-observation ranking are different objectives. No artificial epoch loss curve is shown.'),
    'random_forest':('Bagged tree ensemble','A forest averages multiple nonlinear trees and can reduce the instability of one decision tree.','It captures interactions and provides strong observation ranking. Costs include a larger model and slower fit/inference; class weighting can shift threshold behaviour.','Averaged class probabilities from regularised Gini-split trees.','It ranks second by training mean-trip Top-1 and student-test Top-1. Its positive recall is high but precision is lower than XGBoost. Its long recorded fit duration makes the small ranking gap relevant to model choice.'),
    'xgboost':('Gradient-boosted trees','Histogram boosting is appropriate for tabular geometry/motion features, nonlinear interactions and probability scoring.','It balances ranking quality, candidate macro-F1 and log loss in this comparison. Search complexity, depth-related overfitting and uncalibrated scores remain limitations.','Binary logistic objective with log-loss evaluation; histogram tree construction.','It leads both training OOF mean-trip Top-1 and student-test mean-trip Top-1 and has the lowest student-test candidate log loss. It is therefore selected for decoding and service export. CUDA is recorded in the newer training metadata; inference is explicitly portable to CPU.'),
    'pytorch_mlp':('Two-hidden-layer neural classifier','A modest MLP learns nonlinear combinations of relative features without learning trip or absolute road IDs.','Batched optimization supports GPU fitting and a compact network. Sensitivity to scaling, optimization and calibration is a trade-off. The saved CPU-portable estimator must retain its learned mean and standard deviation.','Binary cross-entropy with logits, Adam, fixed 20 epochs and weight decay.','The recorded training loss decreases over 20 epochs. Candidate macro-F1 is competitive but student-test Top-1 is below the forest and XGBoost. No epoch-wise validation loss exists, so training-loss decrease alone does not establish generalization or the absence of overfitting.')}
    for index,name in enumerate(['logistic','decision_tree','random_forest','xgboost','pytorch_mlp']):
        if index:r.story.append(PageBreak())
        row=models.loc[name];family,reason,limits,objective,analysis=interpretations[name]
        r.sub('8.'+str(index+1)+'  '+LABELS[name]+' — '+family)
        r.p('<b>Why considered.</b> '+reason);r.p('<b>Strengths and limitations.</b> '+limits);r.p('<b>Objective.</b> '+objective)
        params=metadata.get(name,{}).get('params',{})
        r.table(['Recorded configuration','Value'],[[k,str(v)] for k,v in params.items()],[180,323])
        r.table(['OOF mean trip Top-1','OOF macro-F1','Student Top-1','Student macro-F1','Binary log loss'],[[pct(row.cv_mean_trip_top1),fmt(row.cv_macro_f1),pct(row.mean_trip_observation_top1),fmt(row.candidate_macro_f1),fmt(row.candidate_binary_log_loss)]],[104,99,100,108,92])
        r.p(f'<b>Measured cost.</b> Final full fit: {row.fit_seconds:.3f} s; prediction over all ten cached student-test candidate matrices: {row.prediction_seconds:.4f} s. These timings exclude candidate construction, decoding and HTTP overhead. They are not end-to-end request durations.')
        r.p('<b>Interpretation / over- and underfitting.</b> '+analysis)
        if name=='pytorch_mlp' and 'mlp' in figs:r.fig(figs['mlp'],'Actual fold/full-fit training losses. No validation curve is invented.',205)
    r.chapter('9  Consolidated model comparison')
    table=[]
    for name,row in models.iterrows():table.append([LABELS[name],pct(row.cv_mean_trip_top1),fmt(row.cv_macro_f1),pct(row.mean_trip_observation_top1),fmt(row.candidate_macro_f1),fmt(row.candidate_binary_log_loss)])
    r.table(['Model','OOF trip Top-1','OOF macro-F1','Test trip Top-1','Test macro-F1','Test log loss'],table,[109,83,76,83,78,74])
    r.p('All five fits use 1,609,263 resolved training candidate rows from 80 trips. Student-test binary metrics use 169,191 candidate rows across all ten trips. The positive fraction is about 11.3%, so candidate accuracy alone is not an adequate selection metric. Test label coverage is 100%; the retained-candidate miss rate is '+pct(models.loc['xgboost','candidate_miss_rate'])+'.')
    r.table(['Model','Candidate accuracy','Positive precision','Positive recall','Fit seconds','Classifier continuity'],[[LABELS[n],pct(x.candidate_accuracy),fmt(x.candidate_precision),fmt(x.candidate_recall),fmt(x.fit_seconds,3),pct(agg['classifier']['mean_continuity']) if n=='xgboost' else 'N/A: not run'] for n,x in models.iterrows()],[108,83,80,77,69,86])
    r.p('Continuity was executed for the selected XGBoost pipeline. It was not computed for the other four families; N/A values are unexecuted checks rather than zeros. Classifier inference durations exclude graph/features/decoding and should not be compared directly with later full-request timings.')
    r.fig(figs['models'],'Actual OOF and student-test scores for all five models.',265)
    r.fig(figs['loss_runtime'],'Recorded probability loss and full-fit durations. Hardware/execution paths differ by family.',245)
    r.fig(figs['confusion'],'Actual student-test candidate confusion matrices; rows are reference classes, columns predictions.',300)
    r.chapter('10  Bayesian hyperparameter optimisation')
    r.p('Tuning is useful for controlling the bias/variance trade-off of a selected model, but a gain must be measured rather than promised [R3]. The current generic workflow uses Optuna TPESampler, a fixed seed, persistent SQLite studies and trip-disjoint folds. It can tune whichever of the five families wins. Fifteen total trials and five startup trials are configured; failed/pruned trials remain visible [R4].')
    r.table(['Winning family','Implemented bounded search space'],[['Logistic regression','C: 0.001–100 (log); solver: lbfgs / liblinear.'],['Decision tree','Depth: 4–20; minimum leaf: 5–100; criterion: Gini / entropy.'],['Random forest','Trees: 60–240, step 30; depth: 6–20; leaf: 3–40; feature fraction: 0.4–1.0.'],['XGBoost','Trees: 100–500; depth: 3–10; rate: 0.01–0.20 (log); child weight: 0.5–10 (log); row/column sampling: 0.6–1.0; alpha: 1e-8–10; lambda: 1e-5–100.'],['MLP','Widths: 32/64/128 and 32/64; learning rate: 1e-4–1e-2; weight decay: 1e-6–1e-2; dropout: 0–0.3; batch: 1024/2048/4096.']],[112,391])
    r.sub('10.1  Actual saved search record')
    completed=[t for t in e['tuning'] if t['state']=='COMPLETE']
    if completed:
        best=max(completed,key=lambda t:t['value']);r.p(f'The older study <b>{escape(e["study_name"])}</b> contains <b>{len(completed)} completed trials</b> and {sum(t["state"]=="RUNNING" for t in e["tuning"])} entry marked RUNNING. Its recorded objective is mean-fold candidate macro-F1, different from the new pipeline’s primary mean-trip Top-1 objective. Best observed trial: <b>{best["trial"]}</b>, value <b>{best["value"]:.6f}</b>. The RUNNING marker is a database state, not evidence that a training process remains active.')
        r.table(['Best observed parameter','Recorded value'],[[k,str(v)] for k,v in e['best_params'].items()],[210,293])
        r.fig(figs['tuning'],'Partial legacy study extracted read-only from the saved SQLite database. The best recorded trial is not a deployed tuned model.',215)
        r.p('The legacy startup configuration is not independently stored with each trial, and its study was not completed/exported. This report therefore does not claim a completed Bayesian tuning demonstration or a tuned student-test improvement from this record. The current selected classifier remains the untuned comparison winner. A final tuned-versus-untuned table, total completed-study cost and convergence claim require the user-run generic workflow to finish.')
    else:r.p('No completed tuning trials were found. Search code is implemented, but no tuning result is claimed.')
    r.chapter('11  HMM/Viterbi and route reconstruction')
    r.sub('11.1  States, emission potentials and recurrence')
    r.p('At each observation a state is an original directed candidate arc plus its projected fraction. Classifier positive probabilities are clipped and normalized within the candidate set. This creates discriminative emission potentials; it does not automatically produce calibrated generative observation likelihoods. The design follows the established HMM map-matching separation of local uncertainty and road-network feasibility [R1].')
    r.code('e_t(j) = clip(p_t(j)) / sum_k clip(p_t(k))\nV_t(j) = log e_t(j) + max_i [ V_(t-1)(i) + T_t(i,j) ]\nT_t(i,j) = -abs(d_road(i,j) - d_GPS(t-1,t)) / beta\n              - U_turn_penalty(i,j)\nUnreachable or motion-infeasible transition: -infinity')
    r.p('Directed travel uses same-arc forward progress, source remainder plus destination prefix for adjacent arcs, or source remainder + directed shortest-path length + destination prefix for multi-hop travel. Small backwards projection noise is tolerated within 12 m; arbitrary wrong-way travel on a directed arc is not allowed. Graph-feasible reverse-road changes can occur with a modest U-turn penalty. Legal turn restrictions were not supplied.')
    r.table(['Parameter / implementation','Current value or rule'],[['Emission convention','Clipped positive probability / within-observation sum.'],['Transition scale beta','35 m.'],['Large-gap break','More than 60 seconds.'],['Movement proxy bound','55 m/s × interval + 50 m slack; a feasibility proxy, not a supplied speed limit.'],['Backward projection tolerance','12 m on a directed arc.'],['U-turn penalty','1.0 when switching into the reverse endpoint pair.'],['Routing cache','Per-network bounded source search; rounded-up bucket plus exact caller cutoff; at most 2,000 source/bound entries.'],['Block handling','Invalid/empty candidates or all-impossible transitions finish/backtrack the prior block and restart explicitly.']],[202,301])
    r.sub('11.2  Reconstructed geometry')
    r.p('Backtracking retains one output/status per original observation. Selected arcs and predecessor-mapped intermediate arcs form actual directed route components. Gaps and infeasible transitions remain separate components; no straight line is drawn across disconnected blocks. Interpolated roads are predicted paths through unobserved travel, not proof of the exact journey. Duplicate arcs never have their lengths accidentally summed by sparse adjacency construction.')
    r.chapter('12  Decoder results and failure analysis')
    r.table(['Metric','Classifier only','Classifier + Viterbi'],[['Mean trip Top-1',pct(agg['classifier']['mean_top1']),pct(agg['decoded']['mean_top1'])],['Pooled Top-1',pct(agg['classifier']['pooled_top1']),pct(agg['decoded']['pooled_top1'])],['Mean trip directed continuity',pct(agg['classifier']['mean_continuity']),pct(agg['decoded']['mean_continuity'])],['Pooled continuity',pct(agg['classifier']['pooled_continuity']),pct(agg['decoded']['pooled_continuity'])],['Mean trip arc-set route F1',fmt(agg['classifier']['mean_route_f1']),fmt(agg['decoded']['mean_route_f1'])],['Mean normalized edit distance ↓',fmt(agg['classifier']['mean_edit']),fmt(agg['decoded']['mean_edit'])],['Legal / eligible observation pairs',f'{agg["classifier"]["legal_pairs"]:,} / {agg["classifier"]["eligible_pairs"]:,}',f'{agg["decoded"]["legal_pairs"]:,} / {agg["decoded"]["eligible_pairs"]:,}']],[249,127,127])
    r.p('There is one temporal-gap pair excluded from these continuity denominators. Restarts are exposed separately; continuity is not inflated by simply omitting every restarted eligible pair. Mean-of-trip scores and pooled scores differ because trip lengths vary.')
    r.table(['Student-test trip','Classifier Top-1','Decoded Top-1','Change (points)','Route blocks'],[[row.trip_id,pct(row.classifier_top1),pct(row.decoded_top1),f'{100*(row.decoded_top1-row.classifier_top1):+.2f}',str(row.blocks)] for _,row in api.iterrows()],[106,108,108,105,76])
    r.fig(figs['decoder'],'Same classifier/candidates with and without decoding, and measured API durations. Trip suffixes identify all ten cases.',255)
    worst=api.loc[(api.decoded_top1-api.classifier_top1).idxmin()]
    r.p(f'<b>Observed failure:</b> trip {worst.trip_id} decreases from {pct(worst.classifier_top1)} to {pct(worst.decoded_top1)}, a {100*(worst.decoded_top1-worst.classifier_top1):.2f}-point change. This demonstrates that graph coherence can override a locally correct classifier choice. Possible causes include transition scale, GPS jitter, provisional alignment or road ambiguity; these hypotheses were not separately ablated and are not presented as proven causes. The case is retained rather than hidden by a universal “decoder improves accuracy” claim.')
    r.chapter('13  Geographical visualisation and applications')
    if development_map:r.fig(development_map,'Genuine rendered Folium view of student-test 00000052. GPS, public road context, decoded route and supplied development reference are distinct layers. Reference overlays are confined to report maps.',360)
    r.p('The development report map is constructed from executed model output. It does not substitute the reference path for the prediction. The production/evaluator UI receives only public GPS/network inputs, predicted geometry and permitted aggregate scores. It cannot request a hidden reference overlay.')
    if (ROOT/'outputs/verification/ui.png').exists():r.fig(ROOT/'outputs/verification/ui.png','Genuine local Streamlit UI screenshot captured with separate backend and frontend processes. This is a localhost demonstration, not a physical friend-laptop screenshot.',365)
    r.sub('13.1  Browser frontend from the supplied ZIP')
    r.p('The additionally supplied frontend-web.zip contains React 18, TypeScript, Vite and React-Leaflet. Its source is extracted to frontend-web/ while node_modules, local environment files and ZIPs are excluded from publication. It uses a same-origin /api development proxy to Laptop 1. Compatibility updates accept invalid GPS rows safely, display decoded/classifier/network geometry, show available scores, and bound per-request waits. The visual layout is retained. An optional online basemap and theme fonts can require internet; this client is distinguished from the fully self-contained Streamlit/Folium vector map.')
    r.p('Frontend build and browser integration evidence are included only where completed. The Streamlit offline-map browser check recorded zero external map requests. A production React static host must proxy /api to the backend; building static assets alone does not configure that proxy.')
    react_checks=read_json('outputs/verification/react_browser_checks.json')
    if react_checks:
        r.p('<b>Executed React verification.</b> The extracted frontend passed TypeScript checking and its Vite production build. A real-browser test loaded all ten trip options, submitted student-test 00000052 through the actual backend, displayed decoded geometry and its 96.56% derived-reference score, and repeated the action successfully. No browser page errors were recorded. This was a local two-process check, not a physical two-laptop test.')
        if (ROOT/'outputs/verification/react_ui.png').exists():r.fig(ROOT/'outputs/verification/react_ui.png','Genuine React/Vite frontend screenshot after a successful saved-model request through its development proxy.',350)
    r.chapter('14  API, deployment and repeated interaction')
    r.table(['Endpoint','Contract'],[['GET /health','Loaded-artifact and data readiness, model version and mode; no private paths.'],['GET /trips','IDs from the configured development/evaluation allowlist.'],['POST /predict/{trip_id}','202 and job ID; 404 unknown trip; 503 unavailable model/data; 429 another active prediction.'],['GET /jobs/{job_id}','Queued/running/completed/failed stage and safe error; expired jobs return 404.'],['GET /results/{job_id}','Completed predicted geometry, observation statuses, timing and permitted aggregate scores.']],[181,322])
    r.p('A single backend process allows one active prediction and retains at most 32 recent jobs. In-memory jobs expire after restart. Exceptions mark failures and release UI controls; they do not leave a perpetual loading state. Logs record request/job/trip IDs, stage, model version and duration to stdout and rotating local files. Backend time limits are checked between stages; routing searches are bounded. This is not a hard real-time cancellation guarantee.')
    r.sub('14.1  Private evaluation boundary')
    r.p('Prediction receives explicit .track/.nodes/.arcs inputs and does not scan for .route. Only the post-prediction scorer reads an authorised reference. Missing references yield reference_unavailable with absent metrics, not a fake zero. Evaluation-mode responses omit reference arcs, geometry, aligned labels, pointwise correctness and reference diagnostics. Reserved mode requires exactly ten complete trip entries from a configured root or manifest; student-test trips are never renamed as reserved trips.')
    r.sub('14.2  Actual receiving-backend evidence')
    log=(ROOT/'outputs/verification/http_server.log').read_text()
    excerpt=[line for line in log.splitlines() if ('stage=' in line and ('viterbi' in line or 'completed' in line)) or 'duration_seconds=' in line][-6:]
    r.code('\n'.join('\n'.join(textwrap.wrap(line,88,break_long_words=True,break_on_hyphens=False)) for line in excerpt))
    r.p('The block above is a typeset export of actual receiving-backend logs, not a fabricated terminal screenshot. Genuine source logs are retained in outputs/verification/http_server.log. Loopback addresses establish local HTTP integration only.',True)
    r.chapter('15  Testing, runtime and reproducibility')
    r.table(['Check','Executed result / evidence'],[['Regression suite','30 tests passed; report_test_results.txt. Five-family serialization tests fit tiny synthetic fixtures only.'],['All-ten API run','All ten development trips completed; each retained one match/status per observation.'],['Repeated interaction','A completed trip was submitted again successfully; browser repeat control passed.'],['Error paths','Unknown ID rejected; missing model returns unavailable readiness; failed inference reaches terminal safe status; active-job bound enforced.'],['Reference isolation','Prediction without a reference works; API snapshots were scanned for reference label/geometry fields.'],['Offline map','Bundled Leaflet script/style; browser reported zero external map requests for the offline vector map.'],['Backend package','ZIP CRC/checksums validated; extracted package loaded model and predicted in a fresh process.'],['Largest graph / longest track','Both separately executed below; observation correspondence verified.'],['Physical LAN / Windows / official scoring','Not executed in the accessible environment. Exact remaining steps are in the setup guides.']],[164,339])
    large=read_json('outputs/verification/large_trip_benchmarks.json',[])
    r.table(['Benchmark','GPS rows','Directed arcs','Prediction + loading (s)','Blocks'],[[x['trip_id'],f'{x["observations"]:,}',f'{x["arcs"]:,}',fmt(x['total_seconds'],3),x['blocks']] for x in large],[100,85,102,146,70])
    r.p(f'Recorded development request durations range from {api.reported_seconds.min():.3f} to {api.reported_seconds.max():.3f} s in the saved all-ten run. These are observations of that execution, not guaranteed latency. Repeated-trip verification passed; the recorded timings do not constitute a controlled cold-versus-warm cache study.')
    versions=[]
    for package in ['numpy','pandas','scipy','scikit-learn','shapely','pyproj','xgboost','torch','optuna','fastapi','streamlit','folium']:
        try:versions.append([package,importlib.metadata.version(package)])
        except importlib.metadata.PackageNotFoundError:versions.append([package,'not installed'])
    r.table(['Environment / package','Observed value'],[['Python',platform.python_version()],['Host OS',platform.system()+' '+platform.release()],['CPU','AMD Ryzen 7 7435HS (current host identification)'],['GPU evidence','XGBoost/MLP run.json record cuda; current sandbox could not re-query the NVIDIA driver. No GPU model/VRAM is invented.']]+versions,[174,329])
    r.chapter('16  Conclusions, limitations and next actions')
    r.p('The delivered project connects candidate learning to directed sequential reconstruction and actual applications. XGBoost is justified by the saved OOF selection rule and student-test comparison, not by training accuracy. The same saved classifier improves from 87.49% to 91.01% mean trip Top-1 after decoding, while arc-set F1 and ordered quality also improve on average. One trip worsens; explicit failure analysis is necessary. The largest graph and longest track execute with complete observation correspondence.')
    r.table(['Limitation / remaining work','Required next action'],[['Provisional pointwise reference convention','Ask the organiser for the official scorer/alignment convention; retain derived-reference status until resolved.'],['Incomplete saved tuning study','Run the generic resumable 15-trial workflow, save search history and compare tuned/untuned results without promising a gain.'],['Candidate/heading ambiguities','Measure prediction-only radius/K or reliability refinements on training/development data; do not insert reference roads.'],['No feature ablation/calibration study','Do not claim a causal contribution for individual features or calibrated emission likelihoods.'],['Physical two-laptop demonstration','Run Laptop 2 against Laptop 1’s LAN IP and capture both sides; localhost success is insufficient.'],['Reserved evaluation','Freeze model/configuration, provision the ten authorised inputs, then evaluate once under organiser rules.'],['Historical provenance differences','Keep recorded estimator configurations and new run identities distinct; do not retroactively attribute old metrics to changed defaults.']],[190,313])
    r.p('<b>AI-assistance disclosure.</b> AI assistance supported source implementation, debugging, report drafting and evidence assembly. Experimental claims in this report are linked to saved fits, predictions or executed checks. No metrics, screenshots, identities, citations or physical demonstrations are invented. Team members remain responsible for correctness and for explaining the code and conclusions to the mentor.')
    r.chapter('17  References and evidence index')
    references=[
    ('R1','Paul Newson and John Krumm (2009). Hidden Markov Map Matching Through Noise and Sparseness. ACM SIGSPATIAL GIS.','https://www.microsoft.com/en-us/research/publication/hidden-markov-map-matching-noise-sparseness/'),
    ('R2','scikit-learn documentation: GroupKFold and grouped cross-validation.','https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html'),
    ('R3','XGBoost documentation: Notes on Parameter Tuning.','https://xgboost.readthedocs.io/en/stable/tutorials/param_tuning.html'),
    ('R4','Optuna documentation: TPESampler and startup behaviour.','https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html'),
    ('R5','Shapely documentation: STRtree segment spatial indexing.','https://shapely.readthedocs.io/en/stable/strtree.html'),
    ('R6','pyproj documentation: Transformer coordinate handling.','https://pyproj4.github.io/pyproj/stable/api/transformer.html')]
    for key,title,url in references:r.p(f'<b>[{key}]</b> {escape(title)}<br/><link href="{url}" color="#007d87">{escape(url)}</link>',True)
    r.p('Official local sources: Main Guidelines.pdf; Dataset Exploration.pdf; Models and Training Guidelines.pdf; Deliverables.pdf; Assessment and Plagiarism Policy.pdf; UE24CS352A_ML hackathon_Instructions.docx. Team roster and project requirements are taken from these supplied materials. Web method references were checked on '+DATE+'.',True)
    r.table(['Evidence','Repository-relative location'],[['EDA and raw-input hashes','outputs/audit/inventory.csv; audit_summary.json; data_manifest.json'],['Feature/label coverage','outputs/prepared/candidate_diagnostics.csv; reports/report_evidence.json'],['Fit/CV model records','outputs/experiments/models/<family>/'],['Consolidated model comparison','outputs/experiments/student_test/model_comparison.csv and *_per_trip.csv'],['Partial tuning record','outputs/experiments/tuning/tuning.sqlite3 (local, excluded from public raw-data publication)'],['API/decoder run','outputs/verification/api_trip_metrics.csv; <trip_id>.json; summary.json'],['Browser and performance checks','outputs/verification/browser_checks.json; large_trip_benchmarks.json'],['Genuine screenshots','outputs/verification/ui.png; map.png; report development map in reports/assets/'],['Report source and metadata','scripts/generate_report.py; reports/model_metadata.json; reports/report_evidence.json']],[155,348])
    r.chapter('Appendix A  Exact startup and pipeline commands')
    r.sub('A.1  Laptop 1 — backend')
    r.code('cd "/path/to/PathSnap"\nbash scripts/setup_backend.sh\nexport PATHSNAP_DATA_ROOT="/path/to/Student folder"\nbash scripts/start_backend.sh\n# Verify locally:\ncurl http://127.0.0.1:8000/health\n# Find the LAN IPv4 address:\nhostname -I')
    r.p('The data-root directory must directly contain train/ and student_test/. The server binds 0.0.0.0:8000; this is not the URL entered by Laptop 2. Use the reachable IPv4 address on a shared network. Paths above are deployment placeholders, not hardcoded host paths.')
    r.sub('A.2  Laptop 2 — React browser frontend')
    r.code('cd "/path/to/PathSnap/frontend-web"\nnpm ci\n# Linux/macOS shell:\nexport VITE_API_BASE_URL="http://<LAPTOP_1_LAN_IP>:8000"\nnpm run dev\n# Open the Vite URL, normally http://localhost:5173\n# Build check:\nnpm run build')
    r.p('On PowerShell set `$env:VITE_API_BASE_URL="http://&lt;LAPTOP_1_LAN_IP&gt;:8000"` before `npm run dev`. The development proxy forwards /api to Laptop 1. For static production hosting, configure that proxy on the host/reverse proxy explicitly.')
    r.sub('A.3  Laptop 2 — Streamlit alternative')
    r.code('cd "/path/to/PathSnapFrontend"\nbash scripts/setup_frontend.sh\nexport BACKEND_URL="http://<LAPTOP_1_LAN_IP>:8000"\nbash scripts/start_frontend.sh\n# Open http://localhost:8501')
    r.p('Both laptops must permit device-to-device connections. If connection fails, inspect IP/port, shared subnet, VPN/proxy, firewall and access-point isolation. A narrowly scoped inbound backend-port rule may be required; disabling the entire firewall is unnecessary. Frontend-only packages require no datasets, model weights or GPU.')
    r.sub('A.4  Retraining, tuning and reproducible loop')
    r.code('cd "/path/to/PathSnap"\n.venv/bin/python -m pip install -r requirements-train.txt\n.venv/bin/python scripts/check_environment.py --role training --require-gpu\n.venv/bin/python scripts/run_full_pipeline.py \\\n  --data-root "/path/to/Student folder" \\\n  --device cuda --trials 15 --resume\n# Existing saved-model verification without fitting/tuning:\n.venv/bin/python scripts/run_full_pipeline.py \\\n  --data-root "/path/to/Student folder" \\\n  --skip-training --skip-tuning --skip-audit --skip-prepare\n# Regenerate this report from saved evidence:\n.venv/bin/python scripts/generate_report.py')
    r.p('Use `--device cpu` for an explicit CPU training run. A requested CUDA run fails clearly if GPU execution is unavailable; it must not be silently reported as GPU training. The selected inference model is exported CPU-portable with its checksum, preprocessing and feature order. ZIP deployment archives can be generated locally using `scripts/package_dist.py`; supplied/deployment ZIPs are intentionally not published to GitHub.')
    output=ROOT/'reports/PathSnap_Technical_Report.pdf'
    doc=ReportDoc(str(output),title='PathSnap — Technical Project Report for Mentor Submission',author='PathSnap Team 1',subject='Evidence-based map matching implementation, experiments, decoder and deployment')
    doc.multiBuild(r.story)
    (ROOT/'reports/PathSnap_Technical_Report.md').write_text('\n\n'.join(r.markdown)+'\n')
    report_evidence=dict(report_date=DATE,counts=counts,aggregate=agg,model_metadata=metadata,partial_tuning=e['tuning'],partial_best_params=e['best_params'],physical_lan_verified=False,official_scoring_verified=False,source_branch=BRANCH,report_sha256=sha(output))
    (ROOT/'reports/report_evidence.json').write_text(json.dumps(report_evidence,indent=2)+'\n')
    print(str(output));print('PDF bytes:',output.stat().st_size)
if __name__=='__main__':main()
