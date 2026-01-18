import { useState } from "react";
import { Toaster } from "@/app/components/ui/sonner";
import { DashboardHeader } from "@/app/components/DashboardHeader";
import { PatientInputForm } from "@/app/components/PatientInputForm";
import { ResultsDashboard } from "@/app/components/ResultsDashboard";
import { AboutSection } from "@/app/components/AboutSection";
import { motion } from "motion/react";
import { predictLiverDisease } from "@/app/utils/mlModel";
import type { PatientData, PredictionResult } from "@/app/types/patient";
import { HelpSection } from "./components/HelpSection";
import { BulkCSVSection } from "./components/BulkCSVSection";

type ViewMode = "home" | "predict" | "bulk" | "help";

export default function App() {
  const [currentView, setCurrentView] = useState<ViewMode>("home");

  const [isProcessing, setIsProcessing] = useState(false);
  const [predictionResult, setPredictionResult] =
    useState<PredictionResult | null>(null);
  const [currentPatientData, setCurrentPatientData] =
    useState<PatientData | null>(null);

  const handlePredict = async (data: PatientData) => {
    setIsProcessing(true);
    setCurrentPatientData(data);

    // Simulate ML model processing time
    await new Promise((resolve) => setTimeout(resolve, 1500));

    // Get prediction from ML model (currently mock)
    const result = predictLiverDisease(data);
    setPredictionResult(result);

    setIsProcessing(false);

    // Scroll to results on mobile
    setTimeout(() => {
      const resultsElement = document.getElementById("results-section");
      if (resultsElement && window.innerWidth < 1024) {
        resultsElement.scrollIntoView({
          behavior: "smooth",
          block: "start",
        });
      }
    }, 100);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 via-blue-50 to-teal-50">
      <Toaster position="top-right" richColors />

      {/* Header */}
      <DashboardHeader currentView={currentView} onViewChange={setCurrentView} />

      {/* Main Content */}
      <main className="max-w-[1920px] mx-auto">
        {/* HOME */}
        {currentView === "home" && (
          <motion.div
            initial={{ opacity: 0, y: 14 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.35 }}
            className="p-4 sm:p-6"
          >
            {/* Welcome + Options */}
            <div className="max-w-7xl mx-auto">
              <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-6 sm:p-8">
                <div className="flex flex-col gap-2">
                  <h2 className="text-2xl sm:text-3xl font-semibold text-gray-900">
                    Welcome to Liver Disease Prediction System
                  </h2>
                  <p className="text-gray-600">
                    Choose an analysis mode to assess liver disease risk using an
                    ILPD-based ML decision support model.
                  </p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-6">
                  {/* Individual */}
                  <button
                    onClick={() => setCurrentView("predict")}
                    className="text-left rounded-xl border border-teal-200 bg-teal-50 hover:bg-teal-100 transition p-5"
                  >
                    <div className="flex items-start justify-between gap-4">
                      <div>
                        <p className="text-sm font-medium text-teal-700">
                          Mode 1
                        </p>
                        <h3 className="text-xl font-semibold text-gray-900 mt-1">
                          Individual Patient Check
                        </h3>
                        <p className="text-gray-600 mt-2">
                          Enter clinical report values for a single patient and
                          get prediction + contributing factors.
                        </p>
                      </div>
                      <span className="inline-flex items-center justify-center w-12 h-12 rounded-xl bg-white border border-teal-200 text-teal-700 font-semibold">
                        1
                      </span>
                    </div>

                    <div className="mt-4">
                      <span className="inline-flex items-center text-sm font-medium text-teal-700">
                        Start Individual Check →
                      </span>
                    </div>
                  </button>

                  {/* Bulk CSV */}
                  <button
                    onClick={() => setCurrentView("bulk")}
                    className="text-left rounded-xl border border-cyan-200 bg-cyan-50 hover:bg-cyan-100 transition p-5"
                  >
                    <div className="flex items-start justify-between gap-4">
                      <div>
                        <p className="text-sm font-medium text-cyan-700">
                          Mode 2
                        </p>
                        <h3 className="text-xl font-semibold text-gray-900 mt-1">
                          Bulk CSV Analysis
                        </h3>
                        <p className="text-gray-600 mt-2">
                          Upload a CSV file and get predictions for multiple
                          patient records with downloadable results.
                        </p>
                      </div>
                      <span className="inline-flex items-center justify-center w-12 h-12 rounded-xl bg-white border border-cyan-200 text-cyan-700 font-semibold">
                        2
                      </span>
                    </div>

                    <div className="mt-4">
                      <span className="inline-flex items-center text-sm font-medium text-cyan-700">
                        Upload CSV →
                      </span>
                    </div>
                  </button>
                </div>
              </div>

              {/* About Model now on HOME */}
              <div className="mt-6">
                <AboutSection />
              </div>
            </div>
          </motion.div>
        )}

        {/* INDIVIDUAL PREDICTION PAGE */}
        {currentView === "predict" && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 p-4 sm:p-6">
            {/* Left Panel - Input Form */}
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.4 }}
              className="lg:sticky lg:top-[140px] h-fit"
            >
              <PatientInputForm
                onPredict={handlePredict}
                isProcessing={isProcessing}
              />
            </motion.div>

            {/* Right Panel - Results */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.4, delay: 0.1 }}
              id="results-section"
            >
              {isProcessing ? (
                <div className="h-full flex items-center justify-center bg-white rounded-lg border-2 border-teal-200">
                  <div className="text-center p-8">
                    <div className="w-16 h-16 border-4 border-teal-600 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
                    <h3 className="text-xl font-semibold text-gray-900 mb-2">
                      Analyzing Patient Data...
                    </h3>
                    <p className="text-gray-600 mb-4">
                      Machine learning model is processing laboratory values and
                      generating risk assessment
                    </p>
                    <div className="flex items-center justify-center gap-2 text-sm text-gray-500">
                      <span className="w-2 h-2 bg-teal-600 rounded-full animate-pulse"></span>
                      <span className="w-2 h-2 bg-teal-600 rounded-full animate-pulse delay-75"></span>
                      <span className="w-2 h-2 bg-teal-600 rounded-full animate-pulse delay-150"></span>
                    </div>
                  </div>
                </div>
              ) : (
                <ResultsDashboard
                  result={predictionResult}
                  patientData={currentPatientData}
                />
              )}
            </motion.div>
          </div>
        )}

        {/* BULK CSV PAGE */}
        {currentView === "bulk" && (
          <motion.div
            initial={{ opacity: 0, y: 14 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.35 }}
            className="p-4 sm:p-6"
          >
            <BulkCSVSection />
          </motion.div>
        )}

        {/* HELP PAGE */}
        {currentView === "help" && (
          <motion.div
            initial={{ opacity: 0, y: 14 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.35 }}
            className="p-4 sm:p-6"
          >
            <HelpSection />
          </motion.div>
        )}
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-gray-200 mt-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="text-center text-sm text-gray-600">
            <p className="mb-1">
              <strong>Liver Disease Prediction System (ILPD ML)</strong> -
              Clinical Decision Support Tool
            </p>
            <p className="text-xs text-gray-500">
              © 2026 - For demonstration and educational purposes. Not for
              clinical use without proper validation and regulatory approval.
            </p>
            <p className="text-xs text-amber-700 mt-2">
              ⚠️ This tool provides ML-based risk prediction support. It is not
              a medical diagnosis. Final decision must be taken by a qualified
              physician.
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}
