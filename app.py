# app.py — DriveGuard  (CSS eksternal: assets/style_dark.css / style_light.css)
# jalankan: streamlit run app.py

import time, tempfile, os
import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

from config import (CLASS_NAMES, PERCLOS_THRESHOLD, PERCLOS_WINDOW, YAWN_THRESHOLD, STATUS_COLORS)
from core.detector  import FaceROIDetector, preprocess_roi
from core.predictor import DrowsinessPredictor
from core.perclos   import PerclosDetector
from alarm.alarm    import AlarmManager

# ─── load CSS helper ────────────────────────────────────────
def load_css(path):
    p = Path(path)
    if p.exists():
        st.markdown(f"<style>{p.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
    else:
        st.warning(f"CSS tidak ditemukan: {path}")

# ─── page config ────────────────────────────────────────────
st.set_page_config(page_title="DriveGuard", page_icon="🛡️",
                   layout="wide", initial_sidebar_state="expanded")

# ─── session state ──────────────────────────────────────────
for k, v in {
    "running":False,"frame_count":0,"drowsy_count":0,"alarm_count":0,
    "perclos_history":[],"event_log":[],"theme":"dark",
    "val_accuracies":{"InceptionV3":92.0,"MobileNetV2":92.0,"ResNet50V2":95.0}
}.items():
    if k not in st.session_state: st.session_state[k] = v

# ─── inject CSS ─────────────────────────────────────────────
load_css(f"assets/style_{st.session_state['theme']}.css")

# ─── sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Pengaturan")
    st.markdown("**🎨 Tema**")
    c1,c2 = st.columns(2)
    if c1.button("🌙 Dark",  use_container_width=True, type="primary" if st.session_state["theme"]=="dark"  else "secondary"):
        st.session_state["theme"]="dark";  st.rerun()
    if c2.button("☀️ Light", use_container_width=True, type="primary" if st.session_state["theme"]=="light" else "secondary"):
        st.session_state["theme"]="light"; st.rerun()

    st.markdown("---")
    st.markdown("**📥 Sumber Input**")
    input_mode = st.radio("Pilih sumber:", ["📷 Kamera Live","🎬 Upload Video"], index=0)
    uploaded_video=None; camera_index=0
    if input_mode=="🎬 Upload Video":
        uploaded_video=st.file_uploader("Upload video",type=["mp4","avi","mov","mkv"])
        if uploaded_video: st.success(f"✓ {uploaded_video.name}")
        st.caption("💡 Alarm tetap berbunyi saat kantuk terdeteksi.")
    else:
        camera_index=st.number_input("Index kamera",0,5,0,1)

    st.markdown("---")
    st.markdown("**🎚️ Threshold**")
    perclos_th=st.slider("Batas kantuk (%)",50,90,70,5)/100.0
    yawn_th   =st.slider("Batas menguap (frame)",1,5,2)

    st.markdown("---")
    st.markdown("**🏆 Val Accuracy**")
    st.caption("Dari hasil evaluasi Colab")
    ai=st.number_input("InceptionV3 (%)",0.0,100.0,95.0,0.1)
    am=st.number_input("MobileNetV2 (%)",0.0,100.0,93.0,0.1)
    ar=st.number_input("ResNet50V2 (%)", 0.0,100.0,94.0,0.1)
    st.session_state["val_accuracies"]={"InceptionV3":ai,"MobileNetV2":am,"ResNet50V2":ar}

    skip_frames=1; show_all_frames=True
    if input_mode=="🎬 Upload Video":
        st.markdown("---")
        st.markdown("**⏩ Kecepatan**")
        skip_frames    =st.slider("Proses 1 dari N frame",1,10,2)
        show_all_frames=st.checkbox("Tampilkan semua frame",value=True)

    st.markdown("---")
    st.caption("Input→FaceMesh→ROI→3CNN→PERCLOS→Alarm")

# ─── cached system ──────────────────────────────────────────
@st.cache_resource
def load_system(t): return DrowsinessPredictor(val_accuracies=dict(t)), AlarmManager()

# ─── chart colours ──────────────────────────────────────────
def gc():
    if st.session_state["theme"]=="dark":
        return dict(fb="#0d1526",ab="#080c14",tx="#4a6080",li="#00c3ff",
                    da="#ff1744",sp="#1a2840",lb="#0d1526",le="#1a2840",lc="#4a6080")
    return dict(fb="#ffffff",ab="#f5f6f8",tx="#7a8699",li="#0055cc",
                da="#cc1a1a",sp="#e0e4ea",lb="#ffffff",le="#e0e4ea",lc="#7a8699")

# ─── event log ──────────────────────────────────────────────
def add_event(msg,lv="ok"):
    ts=datetime.now().strftime("%H:%M:%S"); log=st.session_state["event_log"]
    log.append({"ts":ts,"msg":msg,"level":lv})
    if len(log)>20: log.pop(0)

def render_log(entries):
    if not entries: return '<div class="event-log" style="font-style:italic">Belum ada event...</div>'
    lines="".join(f'<span class="ev-{e["level"]}">[{e["ts"]}] {e["msg"]}</span><br>' for e in reversed(entries[-12:]))
    return f'<div class="event-log">{lines}</div>'

# ─── overlay ────────────────────────────────────────────────
def draw_overlay(frame,det,lp,rp,mp,ps,pm,fn=None,tf=None):
    frame=frame.copy(); id_=ps.is_drowsy
    bc=(120,20,20) if id_ else (15,80,35)
    ov=frame.copy(); cv2.rectangle(ov,(0,0),(frame.shape[1],52),bc,-1)
    cv2.addWeighted(ov,0.75,frame,0.25,0,frame)
    lc=(80,80,255) if id_ else (40,210,100)
    cv2.putText(frame,("  NGANTUK — WASPADA!" if id_ else "  Normal — Waspada"),
                (14,34),cv2.FONT_HERSHEY_SIMPLEX,0.82,lc,2,cv2.LINE_AA)
    if fn and tf:
        pct=fn/tf; bw=frame.shape[1]
        cv2.rectangle(frame,(0,50),(bw,55),(20,22,28),-1)
        cv2.rectangle(frame,(0,50),(int(bw*pct),55),(0,160,255),-1)
        cv2.putText(frame,f"{fn}/{tf} ({pct*100:.0f}%)",(bw-190,48),
                    cv2.FONT_HERSHEY_SIMPLEX,0.38,(130,150,180),1,cv2.LINE_AA)
    if det.get("face_bbox"):
        x1,y1,x2,y2=det["face_bbox"]; fc2=(60,40,220) if id_ else (20,200,80)
        for cx,cy,dx,dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
            cv2.line(frame,(cx,cy),(cx+dx*20,cy),fc2,2)
            cv2.line(frame,(cx,cy),(cx,cy+dy*20),fc2,2)
    for rk,pred,oc,nc in [("left_eye",lp,(30,200,70),(60,60,240)),
                           ("right_eye",rp,(30,200,70),(60,60,240)),
                           ("mouth",mp,(40,160,240),(40,160,240))]:
        rd=det["rois"].get(rk)
        if rd is None: continue
        _,(rx1,ry1,rx2,ry2)=rd; col=nc if pred=="closed_eye" else oc
        cv2.rectangle(frame,(rx1,ry1),(rx2,ry2),col,1)
        lw=len(pred)*6+10
        cv2.rectangle(frame,(rx1,ry1-16),(rx1+lw,ry1),col,-1)
        cv2.putText(frame,pred,(rx1+4,ry1-4),cv2.FONT_HERSHEY_SIMPLEX,0.38,(255,255,255),1,cv2.LINE_AA)
    h=frame.shape[0]
    cv2.rectangle(frame,(0,h-54),(frame.shape[1],h),(10,14,20),-1)
    cv2.line(frame,(0,h-54),(frame.shape[1],h-54),(30,48,70),1)
    pp=ps.perclos_ratio*100; pc=(80,80,255) if pp>=PERCLOS_THRESHOLD*100 else (0,180,255)
    cv2.putText(frame,f"PERCLOS:{pp:.0f}%  Thr:{PERCLOS_THRESHOLD*100:.0f}%",
                (12,h-34),cv2.FONT_HERSHEY_SIMPLEX,0.46,pc,1,cv2.LINE_AA)
    cv2.putText(frame,f"Menguap:{ps.yawn_count}x  "+
                (f"Alasan:{ps.drowsy_reason}" if id_ else "Status:Waspada"),
                (12,h-14),cv2.FONT_HERSHEY_SIMPLEX,0.44,(120,145,180),1,cv2.LINE_AA)
    fw=frame.shape[1]
    for i,(mn,mi) in enumerate(pm.items()):
        sh=mn.replace("InceptionV3","Inc").replace("MobileNetV2","Mob").replace("ResNet50V2","Res")
        dc2=(80,80,240) if mi["class"]=="closed_eye" else (40,210,100)
        cv2.circle(frame,(fw-205,68+i*20),4,dc2,-1)
        cv2.putText(frame,f"{sh}:{mi['class'][:8]} {mi['confidence']*100:.0f}%",
                    (fw-193,72+i*20),cv2.FONT_HERSHEY_SIMPLEX,0.40,(160,180,210),1,cv2.LINE_AA)
    return frame

# ─── chart ──────────────────────────────────────────────────
def make_chart(history,title="PERCLOS"):
    c=gc(); fig,ax=plt.subplots(figsize=(4,2.0))
    fig.patch.set_facecolor(c["fb"]); ax.set_facecolor(c["ab"])
    x=list(range(len(history))); vals=[v*100 for v in history]; thr=PERCLOS_THRESHOLD*100
    ax.fill_between(x,vals,thr,where=[v>=thr for v in vals],alpha=0.28,color=c["da"],interpolate=True)
    ax.fill_between(x,vals,0,where=[v<thr for v in vals],alpha=0.12,color=c["li"],interpolate=True)
    ax.plot(x,vals,color=c["li"],linewidth=1.8,zorder=5)
    ax.axhline(thr,color=c["da"],linestyle="--",linewidth=1.2,label=f"Thr {thr:.0f}%",alpha=0.85)
    ax.set_ylim(0,105); ax.set_xlim(0,max(len(history),PERCLOS_WINDOW))
    ax.set_ylabel("%",fontsize=8,color=c["tx"]); ax.set_xlabel("Frame",fontsize=8,color=c["tx"])
    ax.tick_params(labelsize=7,colors=c["tx"])
    ax.legend(fontsize=7,loc="upper left",facecolor=c["lb"],edgecolor=c["le"],labelcolor=c["lc"])
    ax.set_title(title,fontsize=9,color=c["tx"],pad=5)
    for sp in ax.spines.values(): sp.set_edgecolor(c["sp"]); sp.set_linewidth(0.7)
    fig.tight_layout(pad=0.8); return fig

# ─── process frame ──────────────────────────────────────────
def process_frame(frame,fd,pred,pd_,al,fn=None,tf=None):
    lp=rp=mp="open_eye"; pm={}; det=fd.detect(frame)
    if det["face_detected"]:
        for rk,vn in [("left_eye","l"),("right_eye","r"),("mouth","m")]:
            rd=det["rois"].get(rk)
            if rd is None: continue
            ri,_=rd; res=pred.predict(preprocess_roi(ri))
            if vn=="l": lp=res["class_name"]; pm=res["per_model"]
            elif vn=="r": rp=res["class_name"]
            else: mp=res["class_name"]
        ps=pd_.update(lp,rp,mp)
    else:
        pd_.reset(); ps=pd_.update("open_eye","open_eye","open_eye")
    ah=None; af=False
    if ps.is_drowsy:
        ah=al.trigger()
        if ah: af=True
    fd2=draw_overlay(frame,det,lp,rp,mp,ps,pm,fn,tf)
    return fd2,{"p_state":ps,"left_pred":lp,"right_pred":rp,"mouth_pred":mp,
                "per_model":pm,"alarm_fired":af,"audio_html":ah,"face_found":det["face_detected"]}

# ─── main layout ────────────────────────────────────────────
st.title("🛡️ DriveGuard — Deteksi Kantuk Pengendara")
ml=(f"📷 Kamera Live" if input_mode=="📷 Kamera Live"
    else f"🎬 {uploaded_video.name if uploaded_video else 'Belum diupload'}")
st.markdown(f'<span class="input-badge">{ml}</span>',unsafe_allow_html=True)
st.caption("InceptionV3 · MobileNetV2 · ResNet50V2  —  MediaPipe FaceMesh + PERCLOS")

cv_,ci_=st.columns([2,1],gap="medium")
with cv_:
    vp=st.empty(); ap_=st.empty(); sp_=st.empty(); pb=st.empty()
with ci_:
    st.markdown('<div class="panel-header">👁 Status ROI</div>',unsafe_allow_html=True)
    ml_=st.empty(); mr_=st.empty(); mm_=st.empty()
    st.markdown('<div class="panel-header">📊 PERCLOS</div>',unsafe_allow_html=True)
    pg=st.empty(); cp=st.empty()
    st.markdown('<div class="panel-header">🤖 Voting Model</div>',unsafe_allow_html=True)
    vot=st.empty()
    st.markdown('<div class="panel-header">📈 Statistik</div>',unsafe_allow_html=True)
    stp=st.empty()
    st.markdown('<div class="panel-header">📋 Event Log</div>',unsafe_allow_html=True)
    logp=st.empty()

cs,ce=st.columns(2)
with cs:
    slbl="▶  Mulai Deteksi" if input_mode=="📷 Kamera Live" else "▶  Mulai Analisis Video"
    sbtn=st.button(slbl,type="primary",use_container_width=True)
with ce:
    stbtn=st.button("⏹  Stop",use_container_width=True)

if sbtn:
    st.session_state.update({"running":True,"frame_count":0,"drowsy_count":0,
                              "alarm_count":0,"perclos_history":[],"event_log":[]})
    add_event("Sesi dimulai","ok")
if stbtn:
    st.session_state["running"]=False; add_event("Sesi dihentikan","warn")

# ─── ui update ──────────────────────────────────────────────
def upd(fd2,state):
    ps=state["p_state"]; lp=state["left_pred"]; rp=state["right_pred"]
    mp=state["mouth_pred"]; pm=state["per_model"]; ah=state["audio_html"]
    dk=st.session_state["theme"]=="dark"
    co=("#00e676" if dk else "#1a7f4b"); cd=("#ff1744" if dk else "#cc1a1a")
    cw=("#ffab00" if dk else "#b76e00"); ca=("#00c3ff" if dk else "#0055cc")

    vp.image(cv2.cvtColor(fd2,cv2.COLOR_BGR2RGB),channels="RGB",use_container_width=True)
    if ah: ap_.markdown(ah,unsafe_allow_html=True)
    if ps.is_drowsy:
        sp_.markdown(f'<div class="status-drowsy">⚠️ NGANTUK — {ps.drowsy_reason}</div>',unsafe_allow_html=True)
    else:
        sp_.markdown('<div class="status-normal">✅ Normal — Pengemudi Waspada</div>',unsafe_allow_html=True)

    def ecard(pred,lbl):
        if pred=="closed_eye": cls,ic,vc="metric-danger","🔴",cd
        else:                  cls,ic,vc="metric-ok","🟢",co
        return (f'<div class="metric-card {cls}"><span class="icon">{ic}</span>'
                f'<div><div class="label">{lbl}</div>'
                f'<div class="value" style="color:{vc}">{pred}</div></div></div>')

    ml_.markdown(ecard(lp,"Mata Kiri"),unsafe_allow_html=True)
    mr_.markdown(ecard(rp,"Mata Kanan"),unsafe_allow_html=True)
    if mp=="yawn":
        mm_.markdown(f'<div class="metric-card metric-warn"><span class="icon">🟠</span>'
                     f'<div><div class="label">Mulut</div>'
                     f'<div class="value" style="color:{cw}">yawn</div></div></div>',unsafe_allow_html=True)
    else:
        mm_.markdown(f'<div class="metric-card metric-ok"><span class="icon">🟢</span>'
                     f'<div><div class="label">Mulut</div>'
                     f'<div class="value" style="color:{co}">normal</div></div></div>',unsafe_allow_html=True)

    pv=ps.perclos_ratio*100; gc2=cd if pv>=PERCLOS_THRESHOLD*100 else ca
    pg.markdown(f'<div class="perclos-label" style="color:{gc2}">{pv:.0f}%</div>'
                f'<div class="perclos-sub">PERCLOS ratio</div>',unsafe_allow_html=True)
    hist=list(perclos_det.perclos_history)
    if len(hist)>1: cp.pyplot(make_chart(hist),clear_figure=True)

    if pm:
        h2=""
        for mn,mi in pm.items():
            cf=mi["confidence"]*100; isd=mi["class"]=="closed_eye"
            cc2="danger" if isd else "ok"; ic2="⚠" if isd else "✓"
            sh2=(mn.replace("InceptionV3","Inception V3")
                   .replace("MobileNetV2","MobileNet V2")
                   .replace("ResNet50V2","ResNet50 V2"))
            h2+=(f'<div class="vote-card {cc2}"><div class="vote-model">{sh2}</div>'
                 f'<div class="vote-result">{ic2} {mi["class"]} '
                 f'<span style="font-size:0.78rem;opacity:0.6">{cf:.0f}%</span></div>'
                 f'<div class="conf-bar-bg"><div class="conf-bar-fill {cc2}" style="width:{cf:.0f}%"></div></div>'
                 f'<div class="conf-label">{cf:.1f}% confidence</div></div>')
        vot.markdown(f'<div class="vote-grid">{h2}</div>',unsafe_allow_html=True)

    fc2=st.session_state["frame_count"]; dc2=st.session_state["drowsy_count"]
    ac2=st.session_state["alarm_count"]; pc2=dc2/max(fc2,1)*100
    stp.markdown(
        f'<div class="stats-grid">'
        f'<div class="stat-item"><div class="stat-val">{fc2}</div><div class="stat-label">Frame</div></div>'
        f'<div class="stat-item"><div class="stat-val">{dc2}</div><div class="stat-label">Kantuk</div></div>'
        f'<div class="stat-item"><div class="stat-val">{pc2:.1f}%</div><div class="stat-label">Rasio</div></div>'
        f'<div class="stat-item"><div class="stat-val">{ac2}</div><div class="stat-label">Alarm</div></div>'
        f'</div>',unsafe_allow_html=True)
    logp.markdown(render_log(st.session_state["event_log"]),unsafe_allow_html=True)

# ─── main loop ──────────────────────────────────────────────
if st.session_state["running"]:
    predictor,alarm=load_system(tuple(sorted(st.session_state["val_accuracies"].items())))
    face_detector=FaceROIDetector()
    perclos_det=PerclosDetector(perclos_th=perclos_th,yawn_th=yawn_th)
    prev_drowsy=False

    if input_mode=="📷 Kamera Live":
        cap=cv2.VideoCapture(int(camera_index),cv2.CAP_DSHOW)
        if not cap.isOpened(): cap=cv2.VideoCapture(int(camera_index))
        if not cap.isOpened():
            st.error("❌ Kamera tidak dapat dibuka."); st.session_state["running"]=False
        else:
            cap.set(cv2.CAP_PROP_BUFFERSIZE,1); st.toast("✓ Kamera aktif",icon="📷")
            add_event("Kamera aktif","ok")
            while st.session_state["running"]:
                ret,frame=cap.read()
                if not ret: st.warning("Frame tidak terbaca."); break
                frame=cv2.flip(frame,1)
                fd2,state=process_frame(frame,face_detector,predictor,perclos_det,alarm)
                st.session_state["frame_count"]+=1; fc2=st.session_state["frame_count"]
                isd=state["p_state"].is_drowsy
                if isd: st.session_state["drowsy_count"]+=1
                if state["alarm_fired"]: st.session_state["alarm_count"]+=1
                if isd and not prev_drowsy: add_event(f"KANTUK—{state['p_state'].drowsy_reason}","danger")
                elif not isd and prev_drowsy: add_event("Kembali normal","ok")
                if not state["face_found"] and fc2%30==0: add_event("Wajah tidak terdeteksi","warn")
                prev_drowsy=isd; upd(fd2,state); time.sleep(0.03)
            cap.release(); face_detector.close()

    elif input_mode=="🎬 Upload Video":
        if uploaded_video is None:
            st.warning("⚠️ Belum ada file video."); st.session_state["running"]=False
        else:
            sfx=os.path.splitext(uploaded_video.name)[-1]
            with tempfile.NamedTemporaryFile(delete=False,suffix=sfx) as tmp:
                tmp.write(uploaded_video.read()); tp=tmp.name
            cap=cv2.VideoCapture(tp)
            if not cap.isOpened():
                st.error("❌ File tidak bisa dibuka."); st.session_state["running"]=False; os.unlink(tp)
            else:
                tf=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps=cap.get(cv2.CAP_PROP_FPS) or 30
                st.toast(f"✓ {tf} frame ({tf/fps:.1f}s)",icon="🎬"); add_event(f"Video: {uploaded_video.name}","ok")
                prog=pb.progress(0.0,text="Menganalisis..."); fi=0; lf=None
                while st.session_state["running"]:
                    ret,frame=cap.read()
                    if not ret: break
                    fi+=1; pct=fi/max(tf,1)
                    prog.progress(min(pct,1.0),text=f"Frame {fi}/{tf} ({pct*100:.0f}%)")
                    if fi%skip_frames!=0:
                        if show_all_frames and lf is not None:
                            vp.image(cv2.cvtColor(lf,cv2.COLOR_BGR2RGB),channels="RGB",use_container_width=True)
                        continue
                    fd2,state=process_frame(frame,face_detector,predictor,perclos_det,alarm,fi,tf)
                    lf=fd2; st.session_state["frame_count"]+=1; isd=state["p_state"].is_drowsy
                    if isd: st.session_state["drowsy_count"]+=1
                    if state["alarm_fired"]: st.session_state["alarm_count"]+=1
                    if isd and not prev_drowsy: add_event(f"Frame {fi}:KANTUK","danger")
                    elif not isd and prev_drowsy: add_event(f"Frame {fi}:Normal","ok")
                    prev_drowsy=isd
                    st.session_state["perclos_history"].append(state["p_state"].perclos_ratio)
                    upd(fd2,state)
                cap.release(); face_detector.close(); os.unlink(tp)
                prog.progress(1.0,text="✅ Selesai!"); st.session_state["running"]=False
                add_event("Analisis selesai","ok")

                st.markdown("---"); st.markdown("## 📊 Hasil Analisis Video")
                fc2=st.session_state["frame_count"]; dc2=st.session_state["drowsy_count"]
                ac2=st.session_state["alarm_count"]; r=dc2/max(fc2,1)
                _c1,_c2,_c3,_c4=st.columns(4)
                _c1.metric("Frame",fc2); _c2.metric("Kantuk",dc2)
                _c3.metric("Rasio",f"{r*100:.1f}%"); _c4.metric("Alarm",ac2)
                if r>0.3: st.error(f"⚠️ Kantuk signifikan ({r*100:.1f}%). Berisiko tinggi!")
                elif r>0.1: st.warning(f"🟡 Kantuk {r*100:.1f}% frame.")
                else: st.success("✅ Pengemudi waspada sepanjang video.")

                hist=st.session_state.get("perclos_history",[])
                if len(hist)>5:
                    st.markdown("### 📈 PERCLOS Keseluruhan")
                    c_=gc(); fig,ax=plt.subplots(figsize=(10,3))
                    fig.patch.set_facecolor(c_["fb"]); ax.set_facecolor(c_["ab"])
                    x_=range(len(hist)); vs=[v*100 for v in hist]; thr_=PERCLOS_THRESHOLD*100
                    ax.fill_between(x_,vs,thr_,where=[v>=thr_ for v in vs],alpha=0.3,color=c_["da"],interpolate=True)
                    ax.fill_between(x_,vs,0,where=[v<thr_ for v in vs],alpha=0.12,color=c_["li"],interpolate=True)
                    ax.plot(vs,color=c_["li"],linewidth=1.5)
                    ax.axhline(thr_,color=c_["da"],linestyle="--",linewidth=1.2,label=f"Thr {thr_:.0f}%")
                    ax.set_ylim(0,105); ax.set_ylabel("PERCLOS (%)",color=c_["tx"])
                    ax.set_xlabel("Frame",color=c_["tx"]); ax.tick_params(colors=c_["tx"])
                    ax.legend(facecolor=c_["lb"],edgecolor=c_["le"],labelcolor=c_["lc"])
                    for sp_ in ax.spines.values(): sp_.set_edgecolor(c_["sp"])
                    fig.tight_layout(); st.pyplot(fig)

else:
    dk=st.session_state["theme"]=="dark"
    is_=('filter:drop-shadow(0 0 14px rgba(0,195,255,0.45));' if dk else '')
    vp.markdown(
        f'<div class="idle-placeholder">'
        f'<span class="idle-icon" style="{is_}">🛡️</span>'
        f'<div class="idle-title">DriveGuard siap</div>'
        f'<div class="idle-sub">Klik <b>▶ Mulai Deteksi</b> untuk memulai</div>'
        f'</div>',unsafe_allow_html=True)
    sp_.markdown('<div class="status-warning">⏳ Menunggu input...</div>',unsafe_allow_html=True)
    logp.markdown(render_log(st.session_state["event_log"]),unsafe_allow_html=True)