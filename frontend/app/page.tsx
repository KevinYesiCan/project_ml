"use client";
import { useState, useMemo, useEffect } from "react";
import axios from "axios";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  AreaChart, Area,
  PieChart, Pie, Cell
} from "recharts";

import {
  FiHome, FiUsers, FiPieChart, FiUploadCloud,
  FiCheckCircle, FiAlertTriangle, FiArrowRight,
  FiDownload, FiX, FiActivity, FiFilter, 
  FiCreditCard, FiBarChart2, FiAlignLeft, FiWifi, 
  FiClock, FiCalendar, FiCpu, FiLayers, FiShield, FiZap
} from "react-icons/fi";

// --- CONFIG & COLORS ---
const COLORS = {
  Safe: '#10B981', // Emerald-500
  Risk: '#F43F5E', // Rose-500
  Warning: '#F59E0B' // Amber-500
};

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [contractFilter, setContractFilter] = useState("All");

  // Modal State
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [modalType, setModalType] = useState<'total' | 'risk' | 'safe'>('total');

  // --- CLOCK & DATE STATE ---
  const [currentDate, setCurrentDate] = useState(new Date());

  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentDate(new Date());
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  const formatDate = (date: Date) => {
    return new Intl.DateTimeFormat('en-US', { 
      weekday: 'long', 
      year: 'numeric', 
      month: 'long', 
      day: 'numeric' 
    }).format(date);
  };

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString('en-US', { 
      hour12: false, 
      hour: '2-digit', 
      minute: '2-digit',
      second: '2-digit'
    });
  };

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      // Ensure your backend is running on this port
      const res = await axios.post("http://127.0.0.1:8001/predict", formData);
      setResult(res.data);
    } catch (err) {
      console.error("Error:", err);
      // Fallback mock data for demonstration if server fails
      // alert("Server connection failed. Check console.");
    } finally {
      setLoading(false);
    }
  };

  // --- DATA PROCESSING FOR CHARTS ---
  const extraStats = useMemo(() => {
    if (!result?.details) return { payment: [], internet: [], tenureChart: [] };

    const calculateRisk = (key: string) => {
      const groups: any = {};
      result.details.forEach((c: any) => {
        const val = c[key] || 'Unknown';
        if (!groups[val]) groups[val] = { name: val, total: 0, churn: 0 };
        groups[val].total++;
        if (c.churn_prediction === 1) groups[val].churn++;
      });
      return Object.values(groups).map((g: any) => ({
        ...g,
        rate: ((g.churn / g.total) * 100).toFixed(1)
      })).sort((a: any, b: any) => b.rate - a.rate);
    };

    const paymentStats = calculateRisk('PaymentMethod');
    const internetStats = calculateRisk('InternetService');

    const tenureBins: any = {};
    for(let i=1; i<=72; i+=6) {
        const label = `${i}-${i+5}`;
        tenureBins[label] = { name: label, total: 0, churn: 0, order: i };
    }
    
    result.details.forEach((c: any) => {
        const t = c.tenure;
        for(let i=1; i<=72; i+=6) {
            if(t >= i && t <= i+5) {
                const label = `${i}-${i+5}`;
                if(tenureBins[label]) {
                    tenureBins[label].total++;
                    if(c.churn_prediction === 1) tenureBins[label].churn++;
                }
                break;
            }
        }
    });

    const tenureChart = Object.values(tenureBins)
        .sort((a: any, b: any) => a.order - b.order)
        .map((b: any) => ({
            name: b.name,
            Risk: b.total > 0 ? ((b.churn / b.total) * 100).toFixed(1) : 0,
            Safe: b.total > 0 ? (((b.total - b.churn) / b.total) * 100).toFixed(1) : 0
        }));

    return { payment: paymentStats, internet: internetStats, tenureChart };
  }, [result]);

  const handleExportCSV = (dataToExport: any[]) => {
    if (!dataToExport || dataToExport.length === 0) return;
    const headers = Object.keys(dataToExport[0]).join(",");
    const rows = dataToExport.map(obj => Object.values(obj).join(","));
    const csvContent = "data:text/csv;charset=utf-8," + [headers, ...rows].join("\n");
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", "customer_churn_analysis.csv");
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const getModalData = () => {
    if (!result?.details) return []; 
    switch (modalType) {
      case 'risk': return result.details.filter((c: any) => c.churn_prediction === 1);
      case 'safe': return result.details.filter((c: any) => c.churn_prediction === 0);
      default: return result.details;
    }
  };

  // --- BUSINESS LOGIC VARIABLES ---
  const LOSS_PER_CUSTOMER = 2000;
  const RETENTION_COST = 200;
  
  const estimatedLoss = result?.churn_count * LOSS_PER_CUSTOMER || 0;
  const retentionBudget = result?.churn_count * RETENTION_COST || 0;
  const netRisk = estimatedLoss - retentionBudget;
  const safeCount = result ? result.total_customers - result.churn_count : 0;

  const pieData = result
    ? [
        { name: "Safe", value: safeCount },
        { name: "High Risk", value: result.churn_count },
      ]
    : [];

  const uniqueContracts = result
    ? ["All", ...new Set(result.details.map((d: any) => d.Contract))]
    : [];

  const contractRisk = result?.details?.reduce(
    (acc: any, curr: any) => {
      if (!acc[curr.Contract])
        acc[curr.Contract] = { total: 0, churn: 0 };

      acc[curr.Contract].total += 1;
      if (curr.churn_prediction === 1)
        acc[curr.Contract].churn += 1;

      return acc;
    },
    {}
  );

  const riskSummary = contractRisk &&
    Object.entries(contractRisk)
      .map(([key, val]: any) => ({
        contract: key,
        rate: (val.churn / val.total) * 100,
      }))
      .sort((a, b) => b.rate - a.rate)[0];

  const openModal = (type: 'total' | 'risk' | 'safe') => {
    if (!result) return;
    setModalType(type);
    setIsModalOpen(true);
  };

  return (
    <div className="flex min-h-screen bg-slate-50 font-sans text-slate-800 selection:bg-indigo-100 selection:text-indigo-700">
      
      {/* SIDEBAR */}
      <aside className="w-72 bg-white border-r border-slate-200 hidden md:flex flex-col fixed h-full z-20 shadow-sm font-inter">
        <div className="h-24 flex items-center px-8 border-b border-slate-50">
          <div className="flex items-center gap-3 text-indigo-600">
            <div className="w-10 h-10 bg-gradient-to-br from-indigo-600 to-violet-600 rounded-xl flex items-center justify-center text-white font-bold text-xl shadow-lg shadow-indigo-200">C</div>
            <span className="text-2xl font-bold tracking-tight text-slate-900">Churnly</span>
          </div>
        </div>

        <nav className="flex-1 p-6 space-y-2">
          <NavItem icon={<FiHome />} label="Overview" active />
          {result && (
            <>
              <div className="pt-4 pb-2">
                 <p className="px-4 text-xs font-bold text-slate-400 uppercase tracking-wider">Analysis</p>
              </div>
              <NavItem icon={<FiPieChart />} label="Risk Reports" />
              <NavItem icon={<FiUsers />} label="Customers" />
            </>
          )}
        </nav>

        {/* CLOCK WIDGET */}
        <div className="px-6 pb-2">
          <div className="p-5 rounded-2xl bg-slate-900 text-white shadow-xl shadow-slate-200 relative overflow-hidden group">
            <div className="absolute -right-4 -top-4 w-20 h-20 bg-indigo-500 rounded-full blur-2xl opacity-40 group-hover:opacity-60 transition-opacity"></div>
            <div className="absolute -left-4 -bottom-4 w-16 h-16 bg-pink-500 rounded-full blur-2xl opacity-30 group-hover:opacity-50 transition-opacity"></div>
            <div className="relative z-10">
               <div className="flex items-center gap-2 text-indigo-300 text-xs font-bold uppercase tracking-wider mb-2">
                 <FiClock className="animate-pulse" /> Current Time
               </div>
               <div className="text-3xl font-mono font-bold tracking-tight mb-1 text-white">
                 {formatTime(currentDate)}
               </div>
               <div className="flex items-center gap-2 text-slate-400 text-xs font-medium border-t border-slate-700/50 pt-2 mt-2">
                  <FiCalendar /> {formatDate(currentDate)}
               </div>
            </div>
          </div>
        </div>

        <div className="p-6 pt-2 border-t border-slate-50">
          <div className="flex items-center gap-4 p-3 rounded-2xl bg-slate-50 border border-slate-100 hover:bg-white hover:shadow-md hover:border-indigo-100 transition-all cursor-pointer group">
            <div className="w-12 h-12 rounded-full bg-indigo-100 flex items-center justify-center text-indigo-600 font-bold border-2 border-white shadow-sm group-hover:scale-105 transition-transform">TK</div>
            <div>
              <p className="text-sm font-bold text-slate-700 group-hover:text-indigo-700 transition-colors">Thanakorn.A</p>
              <p className="text-xs text-slate-400">Super Admin</p>
            </div>
          </div>
        </div>
      </aside>

      {/* MAIN CONTENT */}
      <main className="flex-1 md:ml-72 p-6 lg:p-12 transition-all duration-300">
        <div className="max-w-7xl mx-auto space-y-10">
          
          {/* HEADER */}
          <header className="flex flex-col md:flex-row justify-between md:items-center gap-6">
            <div>
              <h1 className="text-3xl font-extrabold text-slate-900 tracking-tight">
                {result ? "Analysis Dashboard" : "Welcome Back, Thanakorn"}
              </h1>
              {!result && <p className="text-slate-500 mt-2">Everything is ready for your daily churn analysis.</p>}
            </div>
          </header>

          {/* --- CASE 1: PRE-IMPORT STATE --- */}
          {!result ? (
            <div className="space-y-8 animate-fade-in-up">
              
              {/* 1. UPLOAD CARD */}
              <section className="bg-white rounded-3xl p-1 bg-gradient-to-b from-white to-slate-50 shadow-sm border border-slate-200">
                <div className="p-8 md:p-12 rounded-[20px] border border-slate-100/50 flex flex-col md:flex-row items-center justify-between gap-10">
                  <div className="flex-1 space-y-4">
                    <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-50 text-indigo-700 text-xs font-bold">
                      <FiUploadCloud /> DATA IMPORT
                    </div>
                    <h2 className="text-2xl font-bold text-slate-900">Start New Analysis</h2>
                    <p className="text-slate-500 leading-relaxed max-w-lg">
                      Upload your customer dataset (.csv) to generate a real-time churn risk report. Our AI model will categorize customers by risk level and provide actionable insights automatically.
                    </p>
                  </div>
                  <div className="flex flex-col sm:flex-row items-center gap-4 w-full md:w-auto">
                    <div className="relative w-full sm:w-80 group">
                      <input
                        type="file"
                        accept=".csv,.xlsx"
                        onChange={(e) => setFile(e.target.files ? e.target.files[0] : null)}
                        className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
                      />
                      <div className={`flex items-center justify-center gap-4 px-6 py-8 border-2 border-dashed rounded-2xl transition-all duration-300
                        ${file ? 'border-indigo-500 bg-indigo-50/50' : 'border-slate-300 bg-slate-50 group-hover:border-indigo-400 group-hover:bg-white'}`}>
                        <div className={`p-4 rounded-full ${file ? 'bg-indigo-100 text-indigo-600' : 'bg-slate-200 text-slate-500'}`}>
                          <FiUploadCloud className="text-2xl" />
                        </div>
                        <div className="text-left">
                          <p className={`text-sm font-bold ${file ? 'text-indigo-900' : 'text-slate-700'}`}>{file ? file.name : "Click to upload CSV"}</p>
                          <p className="text-xs text-slate-400">{file ? (file.size / 1024).toFixed(2) + " KB" : "Support .csv files"}</p>
                        </div>
                      </div>
                    </div>
                    <button
                      onClick={handleUpload}
                      disabled={loading || !file}
                      className={`w-full sm:w-auto px-8 py-5 rounded-2xl font-bold text-sm transition-all shadow-md flex items-center justify-center gap-3 h-full
                        ${loading || !file ? 'bg-slate-100 text-slate-400 cursor-not-allowed shadow-none' : 'bg-indigo-600 text-white hover:bg-indigo-700 hover:shadow-indigo-500/30 active:scale-95'}`}
                    >
                      {loading ? 'Analyzing...' : <>Run Analysis <FiArrowRight className="text-lg" /></>}
                    </button>
                  </div>
                </div>
              </section>

              {/* 2. SYSTEM CAPABILITIES */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                 <div className="bg-white p-8 rounded-3xl border border-slate-100 shadow-sm hover:shadow-md transition-all hover:-translate-y-1 group">
                    <div className="w-14 h-14 bg-blue-50 rounded-2xl flex items-center justify-center text-blue-600 mb-6 group-hover:bg-blue-600 group-hover:text-white transition-colors">
                       <FiCpu className="text-2xl" />
                    </div>
                    <h3 className="text-lg font-bold text-slate-900 mb-2">AI-Powered Model</h3>
                    <p className="text-slate-500 text-sm leading-relaxed">Uses advanced machine learning algorithms (Random Forest/XGBoost) to predict customer behavior with high accuracy.</p>
                 </div>
                 
                 <div className="bg-white p-8 rounded-3xl border border-slate-100 shadow-sm hover:shadow-md transition-all hover:-translate-y-1 group">
                    <div className="w-14 h-14 bg-purple-50 rounded-2xl flex items-center justify-center text-purple-600 mb-6 group-hover:bg-purple-600 group-hover:text-white transition-colors">
                       <FiLayers className="text-2xl" />
                    </div>
                    <h3 className="text-lg font-bold text-slate-900 mb-2">Deep Segmentation</h3>
                    <p className="text-slate-500 text-sm leading-relaxed">Automatically segments customers based on contract type, payment method, and tenure to find root causes.</p>
                 </div>

                 <div className="bg-white p-8 rounded-3xl border border-slate-100 shadow-sm hover:shadow-md transition-all hover:-translate-y-1 group">
                    <div className="w-14 h-14 bg-emerald-50 rounded-2xl flex items-center justify-center text-emerald-600 mb-6 group-hover:bg-emerald-600 group-hover:text-white transition-colors">
                       <FiShield className="text-2xl" />
                    </div>
                    <h3 className="text-lg font-bold text-slate-900 mb-2">Secure Processing</h3>
                    <p className="text-slate-500 text-sm leading-relaxed">Data is processed locally within the session. No sensitive customer information is stored permanently.</p>
                 </div>
              </div>

              {/* 3. FOOTER INFO */}
              <div className="flex flex-col md:flex-row items-center justify-between p-6 bg-slate-100 rounded-3xl text-sm text-slate-500">
                 <div className="flex items-center gap-2">
                    <FiZap className="text-amber-500" />
                    <span>System Status: <span className="text-emerald-600 font-bold">Operational</span></span>
                 </div>
                 <div className="mt-2 md:mt-0">
                    Last Model Update: <span className="font-bold">2024-05-20 (v3.2.1)</span>
                 </div>
              </div>

            </div>
          ) : (
            
            /* --- CASE 2: RESULT DASHBOARD --- */
            <div className="space-y-8 animate-fade-in-up">
              
              {/* 1. STATS GRID */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                 <StatCard onClick={() => openModal('total')} title="Total Customers" value={result.total_customers.toLocaleString()} icon={<FiUsers className="text-blue-600" />} colorClass="bg-blue-50 text-blue-600" desc="Click to view all" clickable />
                 <StatCard onClick={() => openModal('safe')} title="Retained (Safe)" value={(result.total_customers - result.churn_count).toLocaleString()} icon={<FiCheckCircle className="text-emerald-600" />} colorClass="bg-emerald-50 text-emerald-600" desc="Click to view safe" trend="Positive" clickable />
                 <StatCard onClick={() => openModal('risk')} title="High Risk Customers" value={result.churn_count.toLocaleString()} icon={<FiAlertTriangle className="text-rose-600" />} colorClass="bg-rose-50 text-rose-600" desc="Click to view risk" isDanger clickable />
              </div>

              {/* BUSINESS IMPACT */}
              <div className="bg-white p-8 rounded-3xl shadow-sm border border-slate-200">
                <h3 className="text-lg font-bold mb-6">Business Impact</h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <div className="p-4 bg-rose-50 rounded-xl">
                    <p className="text-sm text-slate-500">Estimated Revenue Loss</p>
                    <p className="text-2xl font-bold text-rose-600">฿{estimatedLoss.toLocaleString()}</p>
                  </div>
                  <div className="p-4 bg-amber-50 rounded-xl">
                    <p className="text-sm text-slate-500">Retention Budget</p>
                    <p className="text-2xl font-bold text-amber-600">฿{retentionBudget.toLocaleString()}</p>
                  </div>
                  <div className="p-4 bg-emerald-50 rounded-xl">
                    <p className="text-sm text-slate-500">Net Risk Exposure</p>
                    <p className="text-2xl font-bold text-emerald-600">฿{netRisk.toLocaleString()}</p>
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* PIE CHART */}
                <div className="bg-white p-8 rounded-3xl shadow-sm border border-slate-200">
                  <h3 className="text-lg font-bold mb-6">Risk Distribution (Pie)</h3>
                  <ResponsiveContainer width="100%" height={300}>
                    <PieChart>
                      <Pie
                        data={pieData}
                        dataKey="value"
                        outerRadius={100}
                        label
                      >
                        <Cell fill="#10B981" />
                        <Cell fill="#F43F5E" />
                      </Pie>
                      <RechartsTooltip />
                    </PieChart>
                  </ResponsiveContainer>
                  {riskSummary && (
                  <div className="bg-indigo-50 p-6 rounded-2xl border border-indigo-100 mt-4">
                    <h3 className="font-bold text-indigo-700 mb-2">AI Insight</h3>
                    <p className="text-slate-700">
                      Customers with <strong>{riskSummary.contract}</strong> contract show the highest churn rate at <strong>{riskSummary.rate.toFixed(1)}%</strong>. Consider targeted retention campaign.
                    </p>
                  </div>
                )}
                </div>

                {/* 2. HORIZONTAL BAR (RISK DISTRIBUTION) */}
                <div className="bg-white p-8 rounded-3xl shadow-sm border border-slate-200 flex flex-col justify-center relative overflow-hidden">
                  <div className="flex justify-between items-end mb-6">
                    <div>
                      <h3 className="text-lg font-bold text-slate-900 flex items-center gap-2"><FiAlignLeft /> Risk Distribution</h3>
                      <p className="text-sm text-slate-400">Total vs Churn Prediction</p>
                    </div>
                    <div className="text-right">
                      <span className="block text-3xl font-black text-rose-500 tracking-tight">{result.churn_rate}%</span>
                      <span className="text-xs font-bold text-slate-400 uppercase">Churn Rate</span>
                    </div>
                  </div>
                  <div className="w-full">
                    <div className="h-8 w-full bg-slate-100 rounded-full flex overflow-hidden relative shadow-inner">
                      <div style={{ width: `${100 - result.churn_rate}%` }} className="h-full bg-emerald-500 relative group transition-all duration-1000 ease-out flex items-center justify-center hover:bg-emerald-400 cursor-help">
                         <div className="opacity-0 group-hover:opacity-100 absolute -top-10 bg-slate-800 text-white text-xs py-1 px-2 rounded transition-opacity whitespace-nowrap pointer-events-none">Safe: {(result.total_customers - result.churn_count).toLocaleString()}</div>
                      </div>
                      <div style={{ width: `${result.churn_rate}%` }} className="h-full bg-rose-500 relative group transition-all duration-1000 ease-out flex items-center justify-center hover:bg-rose-400 cursor-help">
                        <div className="absolute inset-0 opacity-10" style={{ backgroundImage: 'linear-gradient(45deg, #fff 25%, transparent 25%, transparent 50%, #fff 50%, #fff 75%, transparent 75%, transparent)', backgroundSize: '10px 10px' }}></div>
                        <div className="opacity-0 group-hover:opacity-100 absolute -top-10 right-0 bg-slate-800 text-white text-xs py-1 px-2 rounded transition-opacity whitespace-nowrap pointer-events-none">Risk: {result.churn_count.toLocaleString()}</div>
                      </div>
                    </div>
                    <div className="flex justify-between items-center mt-4 text-sm font-medium">
                      <div className="flex items-center gap-2"><div className="w-3 h-3 rounded-full bg-emerald-500"></div><span className="text-slate-600">Retained <span className="text-emerald-600 font-bold ml-1">{((100 - result.churn_rate).toFixed(1))}%</span></span></div>
                      <div className="flex items-center gap-2"><span className="text-slate-600 text-right">High Risk <span className="text-rose-600 font-bold ml-1">{Number(result.churn_rate).toFixed(1)}%</span></span><div className="w-3 h-3 rounded-full bg-rose-500"></div></div>
                    </div>
                  </div>
                </div>
              </div>

              {/* 3. LINE CHART */}
              <div className="bg-white p-8 rounded-3xl shadow-sm border border-slate-200">
                <div className="flex justify-between items-center mb-8">
                  <div>
                    <h3 className="text-lg font-bold text-slate-900 flex items-center gap-2">
                      <FiActivity className="text-indigo-600" /> Churn Risk Trend by Tenure
                    </h3>
                    <p className="text-sm text-slate-400">Visualizing risk probability over customer lifetime (months)</p>
                  </div>
                </div>
                <div className="h-[300px] w-full">
                   <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={extraStats.tenureChart} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                        <defs>
                          <linearGradient id="colorRisk" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#F43F5E" stopOpacity={0.2}/>
                            <stop offset="95%" stopColor="#F43F5E" stopOpacity={0}/>
                          </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                        <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{fill: '#94a3b8', fontSize: 12}} dy={10} />
                        <YAxis axisLine={false} tickLine={false} tick={{fill: '#94a3b8', fontSize: 12}} unit="%" />
                        <RechartsTooltip 
                          contentStyle={{ borderRadius: '12px', border: 'none', boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1)' }}
                        />
                        <Area type="monotone" dataKey="Risk" stroke="#F43F5E" strokeWidth={3} fillOpacity={1} fill="url(#colorRisk)" activeDot={{ r: 8 }} />
                      </AreaChart>
                   </ResponsiveContainer>
                </div>
              </div>

              {/* 4. BREAKDOWN TABLES */}
              <div className="flex justify-between items-center">
                <h3 className="text-xl font-bold text-slate-800">Detailed Breakdowns</h3>
                <select
                  value={contractFilter}
                  onChange={(e) => setContractFilter(e.target.value)}
                  className="border border-slate-200 rounded-xl px-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
                >
                  {uniqueContracts.map((c: any) => (
                    <option key={c} value={c}>
                      {c}
                    </option>
                  ))}
                </select>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                <BreakdownTable title="Contract Type" icon={<FiBarChart2 />} data={result.risk_by_contract} keyName="type" onExport={() => handleExportCSV(result.risk_by_contract)} />
                <BreakdownTable title="Payment Method" icon={<FiCreditCard />} data={extraStats.payment} keyName="name" onExport={() => handleExportCSV(extraStats.payment)} />
                <BreakdownTable title="Internet Service" icon={<FiWifi />} data={extraStats.internet} keyName="name" onExport={() => handleExportCSV(extraStats.internet)} />
              </div>

              {/* Reset Button */}
              <div className="flex justify-center pt-6 pb-12">
                <button onClick={() => { setResult(null); setFile(null); }} className="text-sm text-slate-400 hover:text-indigo-600 underline">
                   Upload a different file
                </button>
              </div>

            </div>
          )}
        </div>
      </main>

      {/* MODAL POPUP */}
      {isModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
          <div className="absolute inset-0 bg-slate-900/60 backdrop-blur-sm transition-opacity" onClick={() => setIsModalOpen(false)}></div>
          <div className="bg-white rounded-2xl shadow-2xl w-full max-w-6xl max-h-[85vh] flex flex-col relative z-10 animate-fade-in-up">
            <div className="px-8 py-6 border-b border-slate-100 flex justify-between items-center bg-slate-50/50 rounded-t-2xl">
              <div className="flex items-center gap-3">
                <div className={`p-2 rounded-lg ${modalType === 'risk' ? 'bg-rose-100 text-rose-600' : modalType === 'safe' ? 'bg-emerald-100 text-emerald-600' : 'bg-blue-100 text-blue-600'}`}>
                  {modalType === 'risk' ? <FiAlertTriangle /> : modalType === 'safe' ? <FiCheckCircle /> : <FiUsers />}
                </div>
                <div>
                  <h3 className="text-xl font-bold text-slate-900 capitalize">{modalType === 'total' ? 'All Customers' : `${modalType} Customers`}</h3>
                  <p className="text-sm text-slate-500">Analysis and prediction results</p>
                </div>
              </div>
              <div className="flex items-center gap-3">
                 <button onClick={() => handleExportCSV(getModalData())} className="flex items-center gap-2 px-4 py-2 bg-white border border-slate-200 rounded-lg text-sm font-bold text-slate-600 hover:text-indigo-600 hover:border-indigo-300 transition-colors"><FiDownload /> Export List</button>
                 <button onClick={() => setIsModalOpen(false)} className="w-8 h-8 flex items-center justify-center rounded-full bg-slate-200 hover:bg-slate-300 text-slate-600 transition-colors"><FiX /></button>
              </div>
            </div>
            <div className="overflow-auto flex-1 p-0">
               {getModalData().length > 0 ? (
                 <table className="w-full text-left">
                   <thead className="bg-slate-50 sticky top-0 z-10 text-xs uppercase text-slate-500 font-semibold shadow-sm">
                     <tr>
                       <th className="px-6 py-4 bg-slate-50 pl-8">Customer / Payment</th>
                       <th className="px-6 py-4 bg-slate-50 text-center">Tenure</th>
                       <th className="px-6 py-4 bg-slate-50">Contract</th>
                       <th className="px-6 py-4 bg-slate-50 text-right">Bill ($)</th> 
                       <th className="px-6 py-4 bg-slate-50 text-right">Risk Score</th>
                       <th className="px-6 py-4 bg-slate-50 text-center pr-8">Status</th>
                     </tr>
                   </thead>
                   <tbody className="divide-y divide-slate-100">
                     {getModalData().map((customer: any, idx: number) => (
                       <tr key={idx} className="hover:bg-indigo-50/40 transition-colors group">
                         <td className="px-6 py-4 pl-8">
                           <div className="font-bold text-slate-700">{customer.customerID || 'N/A'}</div>
                           <div className="text-xs text-slate-400 truncate flex items-center gap-1 mt-1"><FiCreditCard className="inline" /> {customer.PaymentMethod || '-'}</div>
                         </td>
                         <td className="px-6 py-4 text-center">
                           <div className="flex flex-col items-center justify-center">
                             <div className={`inline-flex items-center justify-center w-8 h-8 rounded-full text-xs font-bold mb-1 ${customer.tenure < 12 ? 'bg-amber-100 text-amber-700' : 'bg-slate-100 text-slate-600'}`}>{customer.tenure}</div>
                             <span className="text-[10px] text-slate-400">months</span>
                           </div>
                         </td>
                         <td className="px-6 py-4"><span className="text-sm text-slate-600 font-medium">{customer.Contract || 'N/A'}</span></td>
                         <td className="px-6 py-4 text-right font-mono text-slate-600">{customer.MonthlyCharges ? `$${Number(customer.MonthlyCharges).toFixed(2)}` : '-'}</td>
                         <td className="px-6 py-4 text-right">
                           <div className="flex flex-col items-end">
                             <span className={`text-lg font-black tracking-tight ${customer.churn_prob > 0.5 ? 'text-rose-600' : 'text-emerald-600'}`}>{(customer.churn_prob ? (customer.churn_prob * 100).toFixed(1) : '0.0')}%</span>
                             <span className="text-[10px] text-slate-400 uppercase font-bold">Probability</span>
                           </div>
                         </td>
                         <td className="px-6 py-4 text-center pr-8">
                            <span className={`inline-flex items-center gap-1 px-3 py-1.5 rounded-full text-xs font-bold shadow-sm border ${customer.churn_prediction === 1 ? 'bg-rose-50 text-rose-600 border-rose-100' : 'bg-emerald-50 text-emerald-600 border-emerald-100'}`}>
                              {customer.churn_prediction === 1 ? <FiAlertTriangle /> : <FiCheckCircle />}
                              {customer.churn_prediction === 1 ? 'High Risk' : 'Retained'}
                            </span>
                         </td>
                       </tr>
                     ))}
                   </tbody>
                 </table>
               ) : (
                  <div className="flex flex-col items-center justify-center h-64 text-slate-400">
                    <FiFilter className="text-4xl mb-2" />
                    <p>No customers found for this category.</p>
                  </div>
               )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// --- SUB-COMPONENTS (Added to ensure code works) ---

const NavItem = ({ icon, label, active }: { icon: any, label: string, active?: boolean }) => (
  <div className={`flex items-center gap-4 px-4 py-3 rounded-xl transition-all cursor-pointer ${active ? 'bg-indigo-50 text-indigo-700 font-bold' : 'text-slate-500 hover:bg-slate-50 hover:text-slate-900'}`}>
    <span className="text-xl">{icon}</span>
    <span className="text-sm">{label}</span>
  </div>
);

const StatCard = ({ title, value, icon, colorClass, desc, trend, isDanger, clickable, onClick }: any) => (
  <div onClick={onClick} className={`bg-white p-6 rounded-3xl border border-slate-100 shadow-sm transition-all hover:shadow-md ${clickable ? 'cursor-pointer hover:-translate-y-1 active:scale-95' : ''}`}>
    <div className="flex items-start justify-between mb-4">
      <div className={`p-3 rounded-2xl ${colorClass}`}>{icon}</div>
      {trend && <span className="px-2 py-1 rounded-lg bg-emerald-50 text-emerald-700 text-xs font-bold flex items-center gap-1">+{trend} <FiActivity /></span>}
    </div>
    <div className="space-y-1">
      <h3 className="text-slate-500 text-sm font-medium">{title}</h3>
      <p className={`text-2xl font-black tracking-tight ${isDanger ? 'text-rose-600' : 'text-slate-800'}`}>{value}</p>
    </div>
    {desc && <p className="text-xs text-slate-400 mt-4 pt-4 border-t border-slate-50 flex items-center gap-2"><FiFilter /> {desc}</p>}
  </div>
);

const BreakdownTable = ({ title, icon, data, keyName, onExport }: any) => (
  <div className="bg-white p-6 rounded-3xl border border-slate-200 shadow-sm flex flex-col h-full">
    <div className="flex items-center justify-between mb-6">
      <div className="flex items-center gap-3">
        <div className="p-2 bg-slate-100 text-slate-600 rounded-lg">{icon}</div>
        <h3 className="font-bold text-slate-900">{title}</h3>
      </div>
      <button onClick={onExport} className="text-slate-400 hover:text-indigo-600 transition-colors"><FiDownload /></button>
    </div>
    <div className="space-y-3 flex-1 overflow-auto max-h-64 pr-2">
      {data && data.length > 0 ? data.map((item: any, i: number) => (
        <div key={i} className="flex items-center justify-between p-3 rounded-xl bg-slate-50 border border-slate-100 hover:border-indigo-100 transition-colors">
          <div className="flex flex-col">
            <span className="font-bold text-sm text-slate-700">{item[keyName] || item.contract || item.name}</span>
            <span className="text-xs text-slate-400">{item.total} Customers</span>
          </div>
          <div className="text-right">
             <span className={`block font-bold ${Number(item.rate) > 30 ? 'text-rose-600' : Number(item.rate) > 15 ? 'text-amber-500' : 'text-emerald-600'}`}>{Number(item.rate).toFixed(1)}%</span>
             <span className="text-[10px] text-slate-400 uppercase">Churn</span>
          </div>
        </div>
      )) : <p className="text-center text-slate-400 text-sm py-4">No data available</p>}
    </div>
  </div>
);