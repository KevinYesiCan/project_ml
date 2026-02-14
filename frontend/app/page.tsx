"use client";
import { useState, useMemo } from "react";
import axios from "axios";
import {
  PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend
} from "recharts";
import {
  FiHome, FiUsers, FiPieChart, FiSettings, FiUploadCloud,
  FiCheckCircle, FiAlertTriangle, FiArrowRight, FiSearch, FiBell,
  FiDownload, FiX, FiActivity, FiFilter
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

  // Modal State
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [modalType, setModalType] = useState<'total' | 'risk' | 'safe'>('total');

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await axios.post("http://127.0.0.1:8001/predict", formData);
      setResult(res.data);
    } catch (err) {
      console.error("Error:", err);
      // Mock data for UI testing if backend fails (Optional: Remove in production)
      // alert("เชื่อมต่อ Server ไม่ได้");
    } finally {
      setLoading(false);
    }
  };

  // --- EXPORT FUNCTION ---
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

  // --- FILTER LOGIC FOR POPUP ---
  const getModalData = () => {
    if (!result?.details) return []; // *Backend ต้องส่ง details กลับมา
    switch (modalType) {
      case 'risk': return result.details.filter((c: any) => c.churn_prediction === 1); // สมมติ 1 = churn
      case 'safe': return result.details.filter((c: any) => c.churn_prediction === 0);
      default: return result.details;
    }
  };

  const openModal = (type: 'total' | 'risk' | 'safe') => {
    if (!result) return;
    setModalType(type);
    setIsModalOpen(true);
  };

  // Prepare Chart Data
  const chartData = result ? [
    { name: 'Retained', value: result.total_customers - result.churn_count },
    { name: 'Churn Risk', value: result.churn_count },
  ] : [];

  return (
    <div className="flex min-h-screen bg-slate-50 font-sans text-slate-800 selection:bg-indigo-100 selection:text-indigo-700">
      
      {/* 1. SIDEBAR */}
      <aside className="w-72 bg-white border-r border-slate-200 hidden md:flex flex-col fixed h-full z-20 shadow-sm">
        <div className="h-24 flex items-center px-8 border-b border-slate-50">
          <div className="flex items-center gap-3 text-indigo-600">
            <div className="w-10 h-10 bg-gradient-to-br from-indigo-600 to-violet-600 rounded-xl flex items-center justify-center text-white font-bold text-xl shadow-lg shadow-indigo-200">
              C
            </div>
            <span className="text-2xl font-bold tracking-tight text-slate-900">Churnly</span>
          </div>
        </div>

        <nav className="flex-1 p-6 space-y-2">
          <NavItem icon={<FiHome />} label="Overview" active />
          {/* <NavItem icon={<FiUsers />} label="Customers" />
          <NavItem icon={<FiPieChart />} label="Analytics" />
          <NavItem icon={<FiSettings />} label="Settings" /> */}
        </nav>

        <div className="p-6 border-t border-slate-50">
          <div className="flex items-center gap-4 p-3 rounded-2xl bg-slate-50 border border-slate-100">
            <div className="w-12 h-12 rounded-full bg-indigo-100 flex items-center justify-center text-indigo-600 font-bold border-2 border-white shadow-sm">TK</div>
            <div>
              <p className="text-sm font-bold text-slate-700">Thanakorn.A</p>
              <p className="text-xs text-slate-400">Super Admin</p>
            </div>
          </div>
        </div>
      </aside>

      {/* 2. MAIN CONTENT */}
      <main className="flex-1 md:ml-72 p-6 lg:p-12 transition-all duration-300">
        <div className="max-w-7xl mx-auto space-y-10">
          
          {/* HEADER */}
          <header className="flex flex-col md:flex-row justify-between md:items-center gap-6">
            <div>
              <h1 className="text-3xl font-extrabold text-slate-900 tracking-tight">Dashboard</h1>
            </div>
            <div className="flex items-center gap-4">
            </div>
          </header>

          {/* UPLOAD CARD */}
          <section className="bg-white rounded-3xl p-1 bg-gradient-to-b from-white to-slate-50 shadow-sm border border-slate-200">
            <div className="p-8 md:p-10 rounded-[20px] border border-slate-100/50 flex flex-col md:flex-row items-center justify-between gap-10">
              <div className="flex-1">
                <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-50 text-indigo-700 text-xs font-bold mb-4">
                  <FiUploadCloud /> DATA IMPORT
                </div>
                <h2 className="text-xl font-bold text-slate-900 mb-3">Import Customer Dataset</h2>
                <p className="text-slate-500 text-sm leading-relaxed max-w-lg">
                  Upload your .csv file to generate a churn risk report. Our AI model will categorize customers by risk level and provide actionable insights automatically.
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
                  <div className={`flex items-center justify-center gap-4 px-6 py-5 border-2 border-dashed rounded-2xl transition-all duration-300
                    ${file 
                      ? 'border-indigo-500 bg-indigo-50/50' 
                      : 'border-slate-300 bg-slate-50 group-hover:border-indigo-400 group-hover:bg-white'
                    }`}>
                    <div className={`p-3 rounded-full ${file ? 'bg-indigo-100 text-indigo-600' : 'bg-slate-200 text-slate-500'}`}>
                      <FiUploadCloud className="text-xl" />
                    </div>
                    <div className="text-left">
                      <p className={`text-sm font-bold ${file ? 'text-indigo-900' : 'text-slate-700'}`}>
                        {file ? file.name : "Click to upload CSV"}
                      </p>
                      <p className="text-xs text-slate-400">{file ? (file.size / 1024).toFixed(2) + " KB" : "CSV or Excel file"}</p>
                    </div>
                  </div>
                </div>

                <button
                  onClick={handleUpload}
                  disabled={loading || !file}
                  className={`w-full sm:w-auto px-8 py-5 rounded-2xl font-bold text-sm transition-all shadow-md flex items-center justify-center gap-3
                    ${loading || !file 
                      ? 'bg-slate-100 text-slate-400 cursor-not-allowed shadow-none' 
                      : 'bg-indigo-600 text-white hover:bg-indigo-700 hover:shadow-indigo-500/30 active:scale-95'
                    }`}
                >
                  {loading ? 'Analyzing...' : <>Run Analysis <FiArrowRight className="text-lg" /></>}
                </button>
              </div>
            </div>
          </section>

          {/* REPORT SECTION */}
          {result && (
            <div className="space-y-8 animate-fade-in-up">
              
              {/* STATS GRID (INTERACTIVE) */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                 {/* Card 1: Total - Clickable */}
                 <StatCard 
                   onClick={() => openModal('total')}
                   title="Total Customers" 
                   value={result?.total_customers?.toLocaleString() ?? "0"}
                   icon={<FiUsers className="text-blue-600" />}
                   colorClass="bg-blue-50 text-blue-600"
                   desc="Click to view all customers"
                   clickable
                 />
                 
                 {/* Card 2: Safe - Clickable */}
                 <StatCard 
                   onClick={() => openModal('safe')}
                   title="Retained (Safe)" 
                   value={(result.total_customers - result.churn_count).toLocaleString()} 
                   icon={<FiCheckCircle className="text-emerald-600" />}
                   colorClass="bg-emerald-50 text-emerald-600"
                   desc="Click to view safe list"
                   trend="Positive"
                   clickable
                 />

                 {/* Card 3: Risk - Clickable */}
                 <StatCard 
                   onClick={() => openModal('risk')}
                   title="High Risk Customers" 
                   value={result.churn_count.toLocaleString()} 
                   icon={<FiAlertTriangle className="text-rose-600" />}
                   colorClass="bg-rose-50 text-rose-600"
                   desc="Click to view risk list"
                   isDanger
                   clickable
                 />
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                
                {/* CHART CARD */}
                <div className="bg-white p-8 rounded-3xl shadow-sm border border-slate-200 lg:col-span-1 flex flex-col justify-between relative overflow-hidden">
                  <div className="absolute top-0 right-0 p-4 opacity-10">
                    <FiPieChart className="text-9xl text-slate-800" />
                  </div>
                  <div>
                    <h3 className="text-lg font-bold text-slate-900">Risk Distribution</h3>
                  </div>
                  
                  <div className="h-[300px] w-full relative mt-6">
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie
                          data={chartData}
                          innerRadius={80}
                          outerRadius={100}
                          paddingAngle={5}
                          dataKey="value"
                          cornerRadius={8}
                        >
                          <Cell key="cell-safe" fill={COLORS.Safe} strokeWidth={0} />
                          <Cell key="cell-risk" fill={COLORS.Risk} strokeWidth={0} />
                        </Pie>
                        <Tooltip 
                          contentStyle={{ borderRadius: '16px', border: 'none', boxShadow: '0 10px 25px -5px rgba(0,0,0,0.1)', padding: '12px 16px' }}
                          itemStyle={{ fontWeight: 'bold' }}
                        />
                        <Legend verticalAlign="bottom" height={36} iconType="circle" />
                      </PieChart>
                    </ResponsiveContainer>
                    <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none pb-10">
                        <span className="text-4xl font-black text-slate-900 tracking-tighter">{result.churn_rate}%</span>
                        <span className="text-xs font-bold text-slate-400 uppercase tracking-widest mt-1">Churn Rate</span>
                    </div>
                  </div>
                </div>

                {/* TABLE SECTION */}
                <div className="bg-white rounded-3xl shadow-sm border border-slate-200 lg:col-span-2 flex flex-col">
                  <div className="px-8 py-6 border-b border-slate-100 flex flex-wrap gap-4 justify-between items-center bg-slate-50/50 rounded-t-3xl">
                    <div>
                      <h3 className="font-bold text-slate-900 text-lg">Risk Breakdown by Contract</h3>
                      <p className="text-slate-400 text-sm">Analysis based on contract type</p>
                    </div>
                    
                    {/* ปุ่ม Export CSV สำหรับตารางนี้ (หรือ Export All) */}
                    <button 
                      onClick={() => handleExportCSV(result.risk_by_contract)}
                      className="flex items-center gap-2 px-4 py-2 bg-white border border-slate-200 rounded-lg text-sm font-bold text-slate-600 hover:text-indigo-600 hover:border-indigo-200 transition-colors shadow-sm"
                    >
                      <FiDownload /> Export Summary
                    </button>
                  </div>

                  <div className="overflow-x-auto">
                    <table className="w-full text-left">
                      <thead className="bg-slate-50 text-xs uppercase text-slate-400 font-bold tracking-wider">
                        <tr>
                          <th className="px-8 py-5">Contract Type</th>
                          <th className="px-8 py-5 text-center">Risk Level</th>
                          <th className="px-8 py-5 text-right">Probability</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-50">
                        {result.risk_by_contract.map((item: any, index: number) => (
                          <tr key={index} className="hover:bg-indigo-50/30 transition-colors group">
                            <td className="px-8 py-5 font-bold text-slate-700 group-hover:text-indigo-700">{item.type}</td>
                            <td className="px-8 py-5 text-center">
                              <span className={`inline-flex items-center px-3 py-1 rounded-full text-xs font-bold border
                                ${item.churn_rate > 50 
                                  ? 'bg-rose-50 text-rose-600 border-rose-100' 
                                  : item.churn_rate > 20 
                                    ? 'bg-amber-50 text-amber-600 border-amber-100' 
                                    : 'bg-emerald-50 text-emerald-600 border-emerald-100'
                                }`}>
                                {item.churn_rate > 50 ? 'High Risk' : item.churn_rate > 20 ? 'Medium' : 'Low'}
                              </span>
                            </td>
                            <td className="px-8 py-5 text-right">
                              <div className="flex items-center justify-end gap-3">
                                <div className="w-32 bg-slate-100 rounded-full h-2 overflow-hidden">
                                  <div 
                                    className={`h-full rounded-full transition-all duration-1000 ease-out ${item.churn_rate > 50 ? 'bg-rose-500' : 'bg-emerald-500'}`} 
                                    style={{ width: `${item.churn_rate}%` }}
                                  />
                                </div>
                                <span className="text-sm font-bold text-slate-900 w-10">{item.churn_rate}%</span>
                              </div>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </main>

      {/* 3. MODAL POPUP */}
      {isModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
          <div className="absolute inset-0 bg-slate-900/60 backdrop-blur-sm transition-opacity" onClick={() => setIsModalOpen(false)}></div>
          <div className="bg-white rounded-2xl shadow-2xl w-full max-w-4xl max-h-[85vh] flex flex-col relative z-10 animate-fade-in-up">
            
            {/* Modal Header */}
            <div className="px-8 py-6 border-b border-slate-100 flex justify-between items-center bg-slate-50/50 rounded-t-2xl">
              <div className="flex items-center gap-3">
                <div className={`p-2 rounded-lg ${
                   modalType === 'risk' ? 'bg-rose-100 text-rose-600' :
                   modalType === 'safe' ? 'bg-emerald-100 text-emerald-600' :
                   'bg-blue-100 text-blue-600'
                }`}>
                  {modalType === 'risk' ? <FiAlertTriangle /> : modalType === 'safe' ? <FiCheckCircle /> : <FiUsers />}
                </div>
                <div>
                  <h3 className="text-xl font-bold text-slate-900 capitalize">{modalType === 'total' ? 'All Customers' : `${modalType} Customers`}</h3>
                  <p className="text-sm text-slate-500">List of customers in this category</p>
                </div>
              </div>
              <div className="flex items-center gap-3">
                 <button 
                  onClick={() => handleExportCSV(getModalData())}
                  className="flex items-center gap-2 px-4 py-2 bg-white border border-slate-200 rounded-lg text-sm font-bold text-slate-600 hover:text-indigo-600 hover:border-indigo-300 transition-colors"
                >
                  <FiDownload /> Export List
                </button>
                <button 
                  onClick={() => setIsModalOpen(false)}
                  className="w-8 h-8 flex items-center justify-center rounded-full bg-slate-200 hover:bg-slate-300 text-slate-600 transition-colors"
                >
                  <FiX />
                </button>
              </div>
            </div>

            {/* Modal Content (Table) */}
            <div className="overflow-auto flex-1 p-0">
               {getModalData().length > 0 ? (
                 <table className="w-full text-left">
                   <thead className="bg-slate-50 sticky top-0 z-10 text-xs uppercase text-slate-500 font-semibold">
                     <tr>
                       <th className="px-8 py-4 bg-slate-50">Customer ID</th>
                       <th className="px-8 py-4 bg-slate-50">Contract</th>
                       <th className="px-8 py-4 bg-slate-50 text-right">Risk Score</th>
                       <th className="px-8 py-4 bg-slate-50 text-center">Status</th>
                     </tr>
                   </thead>
                   <tbody className="divide-y divide-slate-100">
                     {getModalData().map((customer: any, idx: number) => (
                       <tr key={idx} className="hover:bg-slate-50/80 transition-colors">
                         <td className="px-8 py-4 font-medium text-slate-700">{customer.customerID || 'N/A'}</td>
                         <td className="px-8 py-4 text-slate-500">{customer.Contract || 'N/A'}</td>
                         <td className="px-8 py-4 text-right font-mono font-bold text-slate-600">
                           {(customer.churn_prob ? (customer.churn_prob * 100).toFixed(1) : '0.0')}%
                         </td>
                         <td className="px-8 py-4 text-center">
                            <span className={`px-2 py-1 rounded text-xs font-bold ${
                              customer.churn_prediction === 1 ? 'bg-rose-100 text-rose-700' : 'bg-emerald-100 text-emerald-700'
                            }`}>
                              {customer.churn_prediction === 1 ? 'Risky' : 'Safe'}
                            </span>
                         </td>
                       </tr>
                     ))}
                   </tbody>
                 </table>
               ) : (
                 <div className="flex flex-col items-center justify-center h-64 text-slate-400">
                   <FiFilter className="text-4xl mb-2 opacity-50" />
                   <p>No customer data available in detail.</p>
                   <p className="text-xs mt-2 text-slate-300">Ensure your backend returns a 'details' array.</p>
                 </div>
               )}
            </div>
            
            {/* Modal Footer */}
            <div className="px-8 py-4 border-t border-slate-100 bg-slate-50 rounded-b-2xl text-right">
              <span className="text-sm text-slate-500">Showing {getModalData().length} records</span>
            </div>

          </div>
        </div>
      )}
    </div>
  );
}

// ─── REUSABLE COMPONENTS ─────────────────────────────

function NavItem({ icon, label, active }: any) {
  return (
    <a href="#" className={`flex items-center gap-4 px-4 py-3.5 rounded-xl text-sm font-bold transition-all
      ${active 
        ? "bg-indigo-600 text-white shadow-lg shadow-indigo-500/30" 
        : "text-slate-500 hover:bg-slate-50 hover:text-indigo-600"
      }`}>
      <span className="text-xl">{icon}</span>
      {label}
    </a>
  );
}

function StatCard({ title, value, icon, desc, isDanger, trend, colorClass, onClick, clickable }: any) {
  return (
    <div 
      onClick={clickable ? onClick : undefined}
      className={`bg-white p-6 rounded-3xl shadow-sm border border-slate-200 flex flex-col justify-between h-full transition-all duration-200 relative overflow-hidden group
      ${clickable ? 'cursor-pointer hover:shadow-lg hover:-translate-y-1 hover:border-indigo-200' : ''}
    `}>
      <div className="flex justify-between items-start mb-4 relative z-10">
        <div className={`p-3 rounded-2xl ${colorClass}`}>
          {icon}
        </div>
        {trend && (
          <span className="text-xs font-bold text-emerald-600 bg-emerald-50 border border-emerald-100 px-2 py-1 rounded-lg flex items-center gap-1">
            <FiActivity /> Safe
          </span>
        )}
      </div>
      <div className="relative z-10">
        <h3 className="text-4xl font-extrabold text-slate-900 tracking-tight">{value}</h3>
        <p className={`text-sm font-bold mt-1 ${isDanger ? 'text-rose-600' : 'text-slate-500'}`}>
          {title}
        </p>
        <div className="flex justify-between items-end mt-4">
           <p className="text-xs text-slate-400 font-medium">{desc}</p>
           {clickable && <FiArrowRight className="text-slate-300 group-hover:text-indigo-500 transition-colors" />}
        </div>
      </div>
      
      {/* Decorative BG Element */}
      <div className={`absolute -right-6 -bottom-6 w-24 h-24 rounded-full opacity-10 blur-2xl transition-transform group-hover:scale-150 ${colorClass.split(" ")[0]}`}></div>
    </div>
  );
}