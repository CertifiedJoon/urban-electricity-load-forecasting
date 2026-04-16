import React, { useEffect } from 'react';
import { motion } from 'framer-motion';
import { Activity, ShieldAlert, Cpu, Zap, TrendingUp, Key, Download, Image as ImageIcon } from 'lucide-react';
import 'katex/dist/katex.min.css';
import { BlockMath } from 'react-katex';
import { usePDF } from 'react-to-pdf';
import html2canvas from 'html2canvas';
import headerImage from './assets/Poster Header.jpg';

const FadeIn = ({ children, delay = 0, className = '', style = {} }) => (
  <motion.div
    initial={{ opacity: 0 }}
    whileInView={{ opacity: 1 }}
    viewport={{ once: true }}
    transition={{ duration: 0.6, delay }}
    className={className}
    style={style}
  >
    {children}
  </motion.div>
);

function App() {
  const { toPDF, targetRef } = usePDF({
    filename: 'academic-poster-800x2090.pdf',
    page: { format: [800, 2090], orientation: 'portrait' },
    canvas: { useCORS: true },
    resolution: 3
  });

  const handleExport = () => {
    document.body.classList.add('exporting-pdf');
    setTimeout(() => {
      toPDF();
      setTimeout(() => document.body.classList.remove('exporting-pdf'), 2000);
    }, 500);
  };

  const handleExportPNG = () => {
    document.body.classList.add('exporting-pdf');
    setTimeout(() => {
      if (targetRef.current) {
        html2canvas(targetRef.current, { scale: 3.69, useCORS: true, backgroundColor: '#000' }).then((canvas) => {
          const imgData = canvas.toDataURL('image/png');
          const link = document.createElement('a');
          link.href = imgData;
          link.download = 'academic-poster.png';
          link.click();
          document.body.classList.remove('exporting-pdf');
        });
      }
    }, 500);
  };

  return (
    <div style={{ display: 'flex', justifyContent: 'center', background: '#000', minHeight: '100vh', padding: '40px 0' }}>

      <div style={{ position: 'fixed', bottom: '40px', right: '40px', zIndex: 9999, display: 'flex', gap: '16px' }}>
        <button
          onClick={handleExportPNG}
          style={{
            background: 'var(--neon-orange)', color: '#000', border: 'none',
            borderRadius: '50%', width: '60px', height: '60px',
            display: 'flex', justifyContent: 'center', alignItems: 'center',
            cursor: 'pointer', boxShadow: '0 4px 12px rgba(249,115,22,0.4)',
          }}
          title="Export high-resolution PNG"
        >
          <ImageIcon size={24} />
        </button>

        <button
          onClick={handleExport}
          style={{
            background: 'var(--neon-blue)', color: '#000', border: 'none',
            borderRadius: '50%', width: '60px', height: '60px',
            display: 'flex', justifyContent: 'center', alignItems: 'center',
            cursor: 'pointer', boxShadow: '0 4px 12px rgba(0,229,255,0.4)',
          }}
          title="Export strictly boxed PDF"
        >
          <Download size={24} />
        </button>
      </div>

      {/* 800x2090 PULL-UP BANNER WRAPPER WITH BLEED PADDING */}
      <div
        ref={targetRef}
        style={{
          width: '800px',
          height: '2090px',
          background: '#000',
          padding: '20px 10px',
          boxSizing: 'border-box',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center'
        }}
      >
        {/* STRICT 780x1470 INNER DESIGN BOX */}
        <div
          className="app-container"
          style={{
            width: '780px',
            height: '1470px',
            background: 'radial-gradient(circle at 40% 30%, #171d2b 0%, #050608 100%)',
            color: 'var(--text-main)',
            position: 'relative',
            overflow: 'hidden',
            padding: '0 32px 24px 32px', // Reset top padding for full-bleed header image
            boxSizing: 'border-box',
            boxShadow: '0 20px 60px rgba(0,229,255,0.1)',
            borderRadius: '12px',
            display: 'flex',
            flexDirection: 'column',
            fontSize: '11px'
          }}
        >
          {/* FULL BLEED HEADER IMAGE (780x150) */}
          <div style={{ margin: '0 -32px 16px -32px', width: '780px', height: '150px', background: '#e53935' }}>
            <img src={headerImage} alt="HKU Header" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
          </div>

          {/* HERO SECTION */}
          <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px', borderBottom: '1px solid rgba(255,255,255,0.1)', paddingBottom: '16px' }}>
            <div style={{ flex: 1 }}>
              <h1 className="title-gradient" style={{ fontSize: '2.1rem', fontWeight: 900, lineHeight: 1.1, margin: 0 }}>
                Probabilistic Electricity Load Forecasting<br />for Urban Buildings
              </h1>
            </div>
            <div style={{ display: 'flex', gap: '24px', textAlign: 'right' }}>
              <div>
                <p style={{ color: 'var(--text-muted)', fontSize: '9px', textTransform: 'uppercase' }}>Author</p>
                <p style={{ color: 'var(--neon-blue)', fontWeight: 600, fontSize: '14px' }}>Joonyoung Moon</p>
              </div>
              <div>
                <p style={{ color: 'var(--text-muted)', fontSize: '9px', textTransform: 'uppercase' }}>Supervisor</p>
                <p style={{ color: 'var(--neon-blue)', fontWeight: 600, fontSize: '14px' }}>Prof. Y. Wang</p>
              </div>
            </div>
          </header>

          {/* BODY ROWS */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '16px', flex: 1 }}>

            {/* SEC 1: PROBLEM */}
            <section>
              <h2 style={{ fontSize: '1.4rem', marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <ShieldAlert color="var(--neon-blue)" size={20} /> The Problem Statement
              </h2>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
                <FadeIn delay={0.1}>
                  <div className="glass-card" style={{ padding: '16px', height: '100%' }}>
                    <h3 style={{ fontSize: '1.1rem', color: 'var(--neon-blue)', marginBottom: '8px' }}>1. The Volatility Problem</h3>
                    <div style={{ lineHeight: 1.5, color: 'var(--text-muted)' }}>
                      <p style={{ marginBottom: '8px' }}>Forecasting at the <strong>individual household level</strong> is severely hindered by highly stochastic occupant behavior and sudden HVAC activity.</p>
                      <p>Systematic underestimation of these localized peaks accelerates thermal transformer degradation. We must definitively shift from deterministic point-estimates toward rigorous probabilistic bounding.</p>
                    </div>
                  </div>
                </FadeIn>

                <FadeIn delay={0.2}>
                  <div className="glass-card" style={{ padding: '16px', height: '100%' }}>
                    <h3 style={{ fontSize: '1.1rem', color: 'var(--neon-blue)', marginBottom: '8px' }}>2. Inadequacy of Pinball Loss</h3>
                    <div style={{ lineHeight: 1.5, color: 'var(--text-muted)' }}>
                      <p style={{ marginBottom: '8px' }}>Interval forecasting natively relies on Quantile (Pinball) Loss to bound uncertainty. However, its standard linear penalty mechanism is structurally inadequate for high-volatility urban loads.</p>
                      <p>Across prolonged multi-step horizons, massive volumes of baseline usage cause <em>temporal dilution</em>. The optimizer is forced into a <strong>"mean-prediction trap,"</strong> systematically averaging out the critical gradients required to anticipate severe capacity spikes.</p>
                    </div>
                  </div>
                </FadeIn>
              </div>
            </section>

            {/* SEC 2: SOLUTION */}
            <section>
              <h2 style={{ fontSize: '1.4rem', marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--neon-orange)' }}>
                <Cpu color="var(--neon-orange)" size={20} /> The Proposed Solution
              </h2>
              <FadeIn delay={0.3}>
                <div className="glass-card orange" bordercolor="var(--neon-orange)" style={{ padding: '16px' }}>
                  <p style={{ fontSize: '12px', lineHeight: 1.5, marginBottom: '12px', color: 'var(--text-muted)' }}>
                    To computationally prioritize event-capture without destroying high-frequency features via smoothing (MVMD), we engineered an algorithm-centric intervention natively integrated with the Temporal Fusion Transformer (TFT).
                  </p>

                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 3fr', gap: '16px', alignItems: 'center' }}>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                      <div>
                        <h4 style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '12px', color: 'var(--neon-orange)' }}><Zap size={14} /> Penalty (Γ)</h4>
                        <p style={{ color: 'var(--text-muted)', lineHeight: 1.4, fontSize: '10px' }}>Applies dynamic extremity scalars strictly against boundary failures.</p>
                      </div>
                      <div>
                        <h4 style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '12px', color: 'var(--neon-orange)' }}><TrendingUp size={14} /> Split-Mean</h4>
                        <p style={{ color: 'var(--text-muted)', lineHeight: 1.4, fontSize: '10px' }}>Independently isolates extreme subset losses to halt dilution.</p>
                      </div>
                      <div>
                        <h4 style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '12px', color: 'var(--neon-orange)' }}><Key size={14} /> Stratification</h4>
                        <p style={{ color: 'var(--text-muted)', lineHeight: 1.4, fontSize: '10px' }}>Forces architectural exposure to rare spikes during data loading.</p>
                      </div>
                    </div>

                    <div className="math-block" style={{ margin: 0, padding: '12px', fontSize: '11px', background: 'rgba(0,0,0,0.8)', border: '1px solid rgba(249, 115, 22, 0.4)' }}>
                      <BlockMath math={"L_{asym}\\left(y,\\hat{y},\\tau\\right)=L_q\\left(y,\\hat{y},\\tau\\right)\\cdot\\Omega\\left(y,\\hat{y},\\tau\\right)"} />
                      <BlockMath math={"\\Omega(y, \\hat{y}, \\tau) = \\begin{cases} \\omega_{peak} & \\text{if } y \\geq \\theta_{peak} \\text{ and } y > \\hat{y} \\text{ and } \\tau = 0.90 \\\\ \\omega_{trough} & \\text{if } y \\leq \\theta_{trough} \\text{ and } y < \\hat{y} \\text{ and } \\tau = 0.10 \\\\ 1 & \\text{otherwise} \\end{cases}"} />
                      <BlockMath math={"L_{split} = \\frac{1}{|S_{ext}|} \\sum_{t \\in S_{ext}} L_{asym}^{(t)} + \\frac{1}{|S_{nom}|} \\sum_{t \\in S_{nom}} L_{asym}^{(t)}"} />
                    </div>
                  </div>
                </div>
              </FadeIn>
            </section>

            {/* SEC 3: RESULTS - EXPANDED */}
            <section style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
              <h2 style={{ fontSize: '1.4rem', marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Activity color="var(--neon-blue)" size={20} /> Empirical Outcomes
              </h2>
              <FadeIn delay={0.4} style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
                <div className="glass-card" style={{ padding: '16px', height: '100%', display: 'flex', flexDirection: 'column', gap: '12px' }}>

                  {/* Visual Metrics Replacement */}
                  <div style={{ display: 'flex', justifyContent: 'space-evenly', alignItems: 'center', background: 'rgba(0,0,0,0.4)', padding: '10px', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.05)' }}>
                    <div style={{ textAlign: 'center' }}>
                      <div style={{ fontSize: '2.5rem', fontWeight: 900, color: 'var(--neon-orange)', textShadow: '0 0 20px rgba(249, 115, 22, 0.4)', lineHeight: 1 }}>+7.8%</div>
                      <div style={{ fontSize: '12px', color: '#fff', fontWeight: 600, marginTop: '4px' }}>Total Capture: 51.4%</div>
                      <div style={{ fontSize: '10px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '1px', marginTop: '2px' }}>P90 Peak Coverage</div>
                    </div>
                    <div style={{ width: '1px', height: '50px', background: 'rgba(255,255,255,0.2)' }}></div>
                    <div style={{ textAlign: 'center' }}>
                      <div style={{ fontSize: '2.5rem', fontWeight: 900, color: 'var(--neon-blue)', textShadow: '0 0 20px rgba(0, 229, 255, 0.4)', lineHeight: 1 }}>98.8%</div>
                      <div style={{ fontSize: '12px', color: '#fff', fontWeight: 600, marginTop: '4px' }}>Maintained Strict Compliance</div>
                      <div style={{ fontSize: '10px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '1px', marginTop: '2px' }}>P10 Trough Coverage</div>
                    </div>
                  </div>

                  <div style={{ background: '#000', padding: '16px', borderRadius: '12px', border: '1px dashed rgba(255,255,255,0.1)', flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    <img src="/assets/benchmark_dashboard.png" alt="Dashboard" style={{ width: '100%', maxHeight: '420px', objectFit: 'contain' }} />
                  </div>
                </div>
              </FadeIn>
            </section>

          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
