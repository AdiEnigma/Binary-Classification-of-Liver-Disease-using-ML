import { useMemo, useState } from "react";
import Papa from "papaparse";
import {
  Upload,
  FileSpreadsheet,
  Download,
  AlertTriangle,
  CheckCircle2,
  Filter,
  ArrowUpDown,
} from "lucide-react";

import { Card, CardContent, CardHeader, CardTitle } from "@/app/components/ui/card";
import { Button } from "@/app/components/ui/button";
import { Badge } from "@/app/components/ui/badge";
import { Input } from "@/app/components/ui/input";

import type { PatientData, PredictionResult } from "@/app/types/patient";
import { predictLiverDisease } from "@/app/utils/mlModel";

type CSVRow = Record<string, string>;

type CanonicalKey =
  | "age"
  | "gender"
  | "totalBilirubin"
  | "directBilirubin"
  | "alkalinePhosphatase"
  | "sgptAlt"
  | "sgotAst"
  | "totalProteins"
  | "albumin"
  | "agRatio"
  | "patientId"
  | "name";

const REQUIRED_CANONICAL: CanonicalKey[] = [
  "age",
  "gender",
  "totalBilirubin",
  "directBilirubin",
  "alkalinePhosphatase",
  "sgptAlt",
  "sgotAst",
  "totalProteins",
  "albumin",
  "agRatio",
];

// ✅ maps any CSV header -> canonical
const HEADER_ALIASES: Record<string, CanonicalKey> = {
  // id/name
  patientid: "patientId",
  patient_id: "patientId",
  id: "patientId",
  name: "name",
  patientname: "name",
  patient_name: "name",

  // age/gender
  age: "age",
  gender: "gender",
  sex: "gender",

  // bilirubin
  totalbilirubin: "totalBilirubin",
  total_bilirubin: "totalBilirubin",
  directbilirubin: "directBilirubin",
  direct_bilirubin: "directBilirubin",

  // alkaline phosphatase
  alkalinephosphatase: "alkalinePhosphatase",
  alkaline_phosphatase: "alkalinePhosphatase",
  alkaline_phosphotase: "alkalinePhosphatase",

  // ALT
  sgptalt: "sgptAlt",
  sgpt_alt: "sgptAlt",
  alt: "sgptAlt",
  alamine_aminotransferase: "sgptAlt",
  alanine_aminotransferase: "sgptAlt",

  // AST
  sgotast: "sgotAst",
  sgot_ast: "sgotAst",
  ast: "sgotAst",
  aspartate_aminotransferase: "sgotAst",

  // proteins
  totalproteins: "totalProteins",
  total_proteins: "totalProteins",
  total_protiens: "totalProteins",

  albumin: "albumin",

  // A/G ratio
  agratio: "agRatio",
  a_g_ratio: "agRatio",
  albumin_and_globulin_ratio: "agRatio",
};

function normalizeKey(k: string) {
  return k.trim().toLowerCase().replace(/\s+/g, "_");
}

function safeNumber(v: any) {
  const n = Number(v);
  return Number.isFinite(n) ? n : 0;
}

function normalizeGender(v: string) {
  const g = (v || "").toString().trim().toLowerCase();
  if (g === "male" || g === "m" || g === "1") return "Male";
  if (g === "female" || g === "f" || g === "0") return "Female";
  return "";
}

function mapToCanonicalRow(row: Record<string, any>) {
  const canonical: Partial<Record<CanonicalKey, string>> = {};

  Object.entries(row).forEach(([key, value]) => {
    const nk = normalizeKey(key);
    const canonicalKey = HEADER_ALIASES[nk];

    if (canonicalKey) {
      canonical[canonicalKey] = (value ?? "").toString();
    }
  });

  return canonical;
}

function canonicalToPatientData(
  canonicalRow: Partial<Record<CanonicalKey, string>>
): PatientData {
  return {
    patientId: canonicalRow.patientId ?? "",
    name: canonicalRow.name ?? "",
    age: safeNumber(canonicalRow.age),
    gender: normalizeGender(canonicalRow.gender ?? ""),
    totalBilirubin: safeNumber(canonicalRow.totalBilirubin),
    directBilirubin: safeNumber(canonicalRow.directBilirubin),
    alkalinePhosphatase: safeNumber(canonicalRow.alkalinePhosphatase),
    sgptAlt: safeNumber(canonicalRow.sgptAlt),
    sgotAst: safeNumber(canonicalRow.sgotAst),
    totalProteins: safeNumber(canonicalRow.totalProteins),
    albumin: safeNumber(canonicalRow.albumin),
    agRatio: safeNumber(canonicalRow.agRatio),
  };
}

type FilterMode = "all" | "risk" | "norisk";
type SortMode = "prob_desc" | "prob_asc";

export function BulkCSVSection() {
  const [rawRows, setRawRows] = useState<Record<string, any>[]>([]);
  const [canonicalRows, setCanonicalRows] = useState<
    Array<Partial<Record<CanonicalKey, string>>>
  >([]);
  const [columns, setColumns] = useState<string[]>([]);
  const [error, setError] = useState<string>("");
  const [fileName, setFileName] = useState<string>("");
  const [isProcessing, setIsProcessing] = useState(false);

  const [results, setResults] = useState<
    Array<{
      input: PatientData;
      output: PredictionResult;
    }>
  >([]);

  // NEW: filter + sort state
  const [filterMode, setFilterMode] = useState<FilterMode>("all");
  const [sortMode, setSortMode] = useState<SortMode>("prob_desc");

  const detectedCanonicalKeys = useMemo(() => {
    const keySet = new Set<CanonicalKey>();
    canonicalRows.forEach((r) => {
      Object.keys(r).forEach((k) => keySet.add(k as CanonicalKey));
    });
    return keySet;
  }, [canonicalRows]);

  const requiredMissing = useMemo(() => {
    return REQUIRED_CANONICAL.filter((k) => !detectedCanonicalKeys.has(k));
  }, [detectedCanonicalKeys]);

  const canRun = canonicalRows.length > 0 && requiredMissing.length === 0;

  const previewRows = useMemo(() => rawRows.slice(0, 10), [rawRows]);

  const handleFile = (file: File) => {
    setError("");
    setResults([]);
    setRawRows([]);
    setCanonicalRows([]);
    setColumns([]);
    setFileName(file.name);

    Papa.parse<Record<string, any>>(file, {
      header: true,
      skipEmptyLines: true,
      complete: (parsed) => {
        const data = parsed.data || [];
        if (!data.length) {
          setError("CSV file is empty or could not be parsed.");
          return;
        }

        setRawRows(data);
        setColumns(Object.keys(data[0] ?? {}));

        const mapped = data.map((row) => mapToCanonicalRow(row));
        setCanonicalRows(mapped);
      },
      error: () => {
        setError("Failed to parse CSV. Please upload a valid CSV file.");
      },
    });
  };

  const runBulkPrediction = async () => {
    if (!canRun) return;
    setIsProcessing(true);
    setError("");

    try {
      const computed = canonicalRows.map((cRow) => {
        const patient = canonicalToPatientData(cRow);
        const pred = predictLiverDisease(patient);
        return { input: patient, output: pred };
      });

      setResults(computed);
    } catch (e) {
      setError("Prediction failed. Check CSV formatting.");
    } finally {
      setIsProcessing(false);
    }
  };

  // ✅ computed filtered + sorted results
  const displayedResults = useMemo(() => {
    let list = [...results];

    if (filterMode === "risk") {
      list = list.filter((r) => r.output.hasDiseaseRisk);
    } else if (filterMode === "norisk") {
      list = list.filter((r) => !r.output.hasDiseaseRisk);
    }

    list.sort((a, b) => {
      const pa = a.output.probability ?? 0;
      const pb = b.output.probability ?? 0;
      return sortMode === "prob_desc" ? pb - pa : pa - pb;
    });

    return list;
  }, [results, filterMode, sortMode]);

  // ✅ stats
  const stats = useMemo(() => {
    if (results.length === 0) return null;
    const risky = results.filter((r) => r.output.hasDiseaseRisk).length;
    const healthy = results.length - risky;
    const avgRisk =
      results.reduce((acc, r) => acc + (r.output.probability || 0), 0) /
      results.length;

    return { risky, healthy, avgRisk: Number(avgRisk.toFixed(1)) };
  }, [results]);

  const downloadResultsCSV = () => {
    if (displayedResults.length === 0) return;

    const exportData = displayedResults.map((r, idx) => ({
      record: idx + 1,
      patientId: r.input.patientId ?? "",
      name: r.input.name ?? "",
      age: r.input.age,
      gender: r.input.gender,
      totalBilirubin: r.input.totalBilirubin,
      directBilirubin: r.input.directBilirubin,
      alkalinePhosphatase: r.input.alkalinePhosphatase,
      sgptAlt: r.input.sgptAlt,
      sgotAst: r.input.sgotAst,
      totalProteins: r.input.totalProteins,
      albumin: r.input.albumin,
      agRatio: r.input.agRatio,
      prediction: r.output.hasDiseaseRisk ? "Disease Risk" : "No Disease Risk",
      probability: r.output.probability,
      confidence: r.output.confidence,
    }));

    const csv = Papa.unparse(exportData);
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);

    const link = document.createElement("a");
    link.href = url;
    link.download = `bulk_predictions_${filterMode}_${sortMode}_${Date.now()}.csv`;
    link.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="max-w-7xl mx-auto">
      <Card className="border border-gray-200 shadow-sm">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-2xl">
            <FileSpreadsheet className="w-6 h-6 text-teal-700" />
            Bulk CSV Analysis
          </CardTitle>
          <p className="text-gray-600 mt-1">
            Upload a CSV file to run predictions on multiple patient records at once.
          </p>
        </CardHeader>

        <CardContent className="p-6 space-y-6">
          {/* Upload */}
          <div className="border-2 border-dashed border-gray-200 rounded-xl p-6 bg-gray-50">
            <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
              <div>
                <p className="font-semibold text-gray-900 flex items-center gap-2">
                  <Upload className="w-5 h-5" />
                  Upload CSV File
                </p>
                {fileName && (
                  <p className="text-xs text-gray-500 mt-2">
                    File loaded: <span className="font-semibold">{fileName}</span>
                  </p>
                )}
              </div>

              <div className="w-full md:w-auto">
                <Input
                  type="file"
                  accept=".csv"
                  onChange={(e) => {
                    const file = e.target.files?.[0];
                    if (file) handleFile(file);
                  }}
                />
              </div>
            </div>
          </div>

          {/* Missing columns warning */}
          {rawRows.length > 0 && requiredMissing.length > 0 && (
            <div className="rounded-xl border border-amber-200 bg-amber-50 p-4 flex gap-3">
              <AlertTriangle className="w-5 h-5 text-amber-800 mt-0.5" />
              <div className="text-sm text-amber-900">
                <p className="font-semibold">Missing required parameters:</p>
                <p className="mt-1">{requiredMissing.join(", ")}</p>
              </div>
            </div>
          )}

          {/* Preview */}
          {rawRows.length > 0 && (
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <p className="font-semibold text-gray-900">Preview (first 10 rows)</p>
                <Badge variant="outline">{rawRows.length} total records</Badge>
              </div>

              <div className="overflow-auto rounded-xl border border-gray-200">
                <table className="min-w-full text-sm">
                  <thead className="bg-gray-50 border-b">
                    <tr>
                      {columns.slice(0, 8).map((c) => (
                        <th
                          key={c}
                          className="text-left p-3 text-xs font-semibold text-gray-700"
                        >
                          {c}
                        </th>
                      ))}
                      <th className="text-left p-3 text-xs font-semibold text-gray-700">
                        …
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {previewRows.map((r, i) => (
                      <tr key={i} className="border-b last:border-b-0">
                        {columns.slice(0, 8).map((c) => (
                          <td key={c} className="p-3 text-gray-700">
                            {(r as any)[c]}
                          </td>
                        ))}
                        <td className="p-3 text-gray-500">…</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Actions */}
          <div className="flex flex-col sm:flex-row sm:items-center gap-3 pt-2">
            <Button
              onClick={runBulkPrediction}
              disabled={!canRun || isProcessing}
              className="bg-teal-600 hover:bg-teal-700 h-11 px-6 w-full sm:w-auto"
            >
              {isProcessing ? "Running Predictions..." : "Run Bulk Prediction"}
            </Button>

            <Button
              variant="outline"
              onClick={downloadResultsCSV}
              disabled={displayedResults.length === 0}
              className="h-11 px-6 w-full sm:w-auto flex items-center gap-2"
            >
              <Download className="w-4 h-4" />
              Download Results CSV
            </Button>
          </div>

          {/* stats */}
          {stats && (
            <div className="rounded-xl border border-green-200 bg-green-50 p-4 flex gap-3">
              <CheckCircle2 className="w-5 h-5 text-green-700 mt-0.5" />
              <div className="text-sm text-green-900">
                <p className="font-semibold">Bulk Analysis Completed</p>
                <p className="mt-1">
                  Risk Detected: <b>{stats.risky}</b> | No Risk: <b>{stats.healthy}</b> | Avg Risk:{" "}
                  <b>{stats.avgRisk}%</b>
                </p>
              </div>
            </div>
          )}

          {/* FILTER + SORT CONTROLS */}
          {results.length > 0 && (
            <div className="flex flex-col md:flex-row gap-3 md:items-center md:justify-between p-4 rounded-xl border bg-white">
              <div className="flex items-center gap-2 text-sm font-semibold text-gray-700">
                <Filter className="w-4 h-4" />
                Results View
              </div>

              <div className="flex flex-col sm:flex-row gap-3">
                {/* Filter */}
                <select
                  value={filterMode}
                  onChange={(e) => setFilterMode(e.target.value as FilterMode)}
                  className="h-10 rounded-lg border border-gray-300 px-3 text-sm"
                >
                  <option value="all">All Records</option>
                  <option value="risk">Disease Risk Only</option>
                  <option value="norisk">No Risk Only</option>
                </select>

                {/* Sort */}
                <select
                  value={sortMode}
                  onChange={(e) => setSortMode(e.target.value as SortMode)}
                  className="h-10 rounded-lg border border-gray-300 px-3 text-sm"
                >
                  <option value="prob_desc">Probability: High → Low</option>
                  <option value="prob_asc">Probability: Low → High</option>
                </select>
              </div>
            </div>
          )}

          {/* RESULTS TABLE PREVIEW */}
          {results.length > 0 && (
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <p className="font-semibold text-gray-900">
                  Results Preview ({displayedResults.length} records shown)
                </p>
              </div>

              <div className="overflow-auto rounded-xl border border-gray-200">
                <table className="min-w-full text-sm">
                  <thead className="bg-gray-50 border-b">
                    <tr>
                      <th className="text-left p-3 text-xs font-semibold text-gray-700">#</th>
                      <th className="text-left p-3 text-xs font-semibold text-gray-700">Age</th>
                      <th className="text-left p-3 text-xs font-semibold text-gray-700">Gender</th>
                      <th className="text-left p-3 text-xs font-semibold text-gray-700">Prediction</th>
                      <th className="text-left p-3 text-xs font-semibold text-gray-700">
                        Probability
                      </th>
                      <th className="text-left p-3 text-xs font-semibold text-gray-700">
                        Confidence
                      </th>
                    </tr>
                  </thead>

                  <tbody>
                    {displayedResults.slice(0, 50).map((r, idx) => (
                      <tr key={idx} className="border-b last:border-b-0">
                        <td className="p-3 text-gray-700">{idx + 1}</td>
                        <td className="p-3 text-gray-700">{r.input.age}</td>
                        <td className="p-3 text-gray-700">{r.input.gender}</td>
                        <td className="p-3">
                          <Badge
                            className={
                              r.output.hasDiseaseRisk
                                ? "bg-red-600 text-white"
                                : "bg-green-600 text-white"
                            }
                          >
                            {r.output.hasDiseaseRisk ? "Disease Risk" : "No Risk"}
                          </Badge>
                        </td>
                        <td className="p-3 text-gray-700 font-semibold">
                          {r.output.probability.toFixed(1)}%
                        </td>
                        <td className="p-3 text-gray-700">{r.output.confidence}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <p className="text-xs text-gray-500">
                Showing first 50 results. Download CSV for complete output.
              </p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
