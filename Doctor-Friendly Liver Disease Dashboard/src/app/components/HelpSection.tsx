import {
    Info,
    TestTube2,
    FileSpreadsheet,
    ShieldAlert,
    Activity,
    HelpCircle,
    ChevronDown,
  } from "lucide-react";
  import { Card, CardContent, CardHeader, CardTitle } from "@/app/components/ui/card";
  import { Badge } from "@/app/components/ui/badge";
  import { Button } from "@/app/components/ui/button";
  import {
    Accordion,
    AccordionContent,
    AccordionItem,
    AccordionTrigger,
  } from "@/app/components/ui/accordion";
  
  type Param = {
    key: string;
    short: string;
    unit: string;
    normal: string;
    why: string;
    high: string[];
    low: string[];
    clinicalHint: string;
  };
  
  const PARAMS: Param[] = [
    {
      key: "TB",
      short: "Total Bilirubin (TB)",
      unit: "mg/dL",
      normal: "0.3 – 1.2",
      why: "Bilirubin is produced when red blood cells break down. The liver processes it and clears it via bile.",
      high: [
        "Liver inflammation (hepatitis)",
        "Bile duct blockage (cholestasis/obstruction)",
        "Hemolysis (excess RBC breakdown)",
      ],
      low: ["Usually not clinically significant"],
      clinicalHint:
        "If TB + DB are both high, bile flow obstruction or liver dysfunction is more likely than hemolysis.",
    },
    {
      key: "DB",
      short: "Direct Bilirubin (DB)",
      unit: "mg/dL",
      normal: "0.0 – 0.3",
      why: "Direct bilirubin is bilirubin that has already been processed by the liver (conjugated).",
      high: [
        "Bile duct obstruction",
        "Cholestatic liver disease",
        "Drug-induced cholestasis",
      ],
      low: ["Usually not clinically significant"],
      clinicalHint:
        "High DB suggests conjugated hyperbilirubinemia → more likely bile duct/liver processing issue.",
    },
    {
      key: "ALP",
      short: "Alkaline Phosphatase (ALP)",
      unit: "IU/L",
      normal: "44 – 147",
      why: "ALP rises when bile flow is impaired. It can also rise in bone disorders (so clinical context matters).",
      high: [
        "Cholestasis / bile duct blockage",
        "Gallstones",
        "Primary biliary conditions",
        "Bone disease (important confounder)",
      ],
      low: ["Rarely clinically significant"],
      clinicalHint:
        "High ALP with high DB points strongly toward a cholestatic pattern.",
    },
    {
      key: "ALT",
      short: "SGPT / ALT",
      unit: "IU/L",
      normal: "7 – 56",
      why: "ALT is an enzyme mainly from liver cells. It rises when liver cells are damaged.",
      high: [
        "Hepatitis (viral, alcoholic, fatty liver)",
        "Drug-induced liver injury",
        "Ischemic liver injury (rare but severe)",
      ],
      low: ["Usually not clinically significant"],
      clinicalHint:
        "ALT is more liver-specific than AST. Very high ALT often indicates hepatocellular injury.",
    },
    {
      key: "AST",
      short: "SGOT / AST",
      unit: "IU/L",
      normal: "10 – 40",
      why: "AST exists in liver and other tissues (muscle/heart). So AST elevation needs context.",
      high: [
        "Hepatitis / liver inflammation",
        "Alcohol-related liver injury (often AST > ALT)",
        "Muscle injury (non-liver cause)",
      ],
      low: ["Usually not clinically significant"],
      clinicalHint:
        "AST:ALT ratio > 2 may suggest alcoholic liver disease (not a diagnosis, just a hint).",
    },
    {
      key: "TP",
      short: "Total Proteins (TP)",
      unit: "g/dL",
      normal: "6.0 – 8.3",
      why: "Total protein includes albumin + globulins. The liver produces albumin.",
      high: [
        "Chronic inflammation",
        "Dehydration",
        "Certain immune-related conditions",
      ],
      low: [
        "Malnutrition",
        "Liver synthetic dysfunction (chronic)",
        "Kidney protein loss",
      ],
      clinicalHint:
        "Low proteins with low albumin can suggest poor nutrition or reduced liver synthetic function.",
    },
    {
      key: "ALB",
      short: "Albumin (ALB)",
      unit: "g/dL",
      normal: "3.5 – 5.0",
      why: "Albumin is made by the liver. Low albumin can indicate reduced liver function over time.",
      high: ["Dehydration (common cause)"],
      low: [
        "Chronic liver disease (reduced synthesis)",
        "Kidney disease (protein loss)",
        "Malnutrition",
      ],
      clinicalHint:
        "Albumin reflects long-term liver function more than acute injury.",
    },
    {
      key: "AGR",
      short: "A/G Ratio",
      unit: "",
      normal: "1.0 – 2.5",
      why: "Ratio of Albumin to Globulin. Globulins often rise in chronic inflammation or liver disease.",
      high: [
        "Low globulins (rare)",
        "Certain genetic/protein disorders",
      ],
      low: [
        "Chronic liver disease",
        "Autoimmune/inflammatory conditions",
        "Infections",
      ],
      clinicalHint:
        "Low A/G ratio can suggest chronic inflammatory state or liver dysfunction.",
    },
  ];
  
  const PATTERN_GUIDE = [
    {
      title: "Hepatocellular Pattern",
      badge: "ALT/AST high",
      desc:
        "Liver cell injury pattern. Often seen in hepatitis, fatty liver disease, drug-induced injury.",
      hint:
        "ALT and AST rise more than ALP. If ALT is much higher than ALP → hepatocellular.",
    },
    {
      title: "Cholestatic Pattern",
      badge: "ALP + DB high",
      desc:
        "Bile flow obstruction pattern. Seen in gallstones, bile duct obstruction, cholestatic diseases.",
      hint:
        "ALP and Direct Bilirubin rise prominently. If ALP dominates → cholestatic.",
    },
    {
      title: "Mixed Pattern",
      badge: "Both elevated",
      desc:
        "Features of both injury + obstruction. Clinical correlation required.",
      hint:
        "ALT/AST and ALP both abnormal in similar severity.",
    },
  ];
  
  export function HelpSection() {
    return (
      <div className="max-w-7xl mx-auto">
        <Card className="bg-white rounded-xl border border-gray-200 shadow-sm">
          <CardHeader className="pb-2">
            <CardTitle className="flex items-center gap-2 text-2xl">
              <HelpCircle className="w-6 h-6 text-teal-700" />
              Help & Lab Parameter Insights
            </CardTitle>
            <p className="text-gray-600 mt-1">
              Understand what each clinical parameter means and how it relates to liver disease risk prediction.
            </p>
  
            <div className="mt-4 rounded-xl border border-amber-200 bg-amber-50 p-4 flex gap-3">
              <ShieldAlert className="w-5 h-5 text-amber-700 mt-0.5" />
              <div className="text-sm text-amber-900">
                <p className="font-semibold">Important Note</p>
                <p className="mt-1">
                  This system provides ML-based risk prediction support only. It does not confirm the exact type of liver disease.
                  Final clinical diagnosis must be made by qualified medical professionals.
                </p>
              </div>
            </div>
          </CardHeader>
  
          <CardContent className="p-6 space-y-8">
            {/* Parameter Cards */}
            <section>
              <div className="flex items-center gap-2 mb-4">
                <TestTube2 className="w-5 h-5 text-gray-700" />
                <h3 className="text-lg font-semibold text-gray-900">Lab Parameters Explained</h3>
                <Badge variant="outline" className="ml-2">Doctor-friendly</Badge>
              </div>
  
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {PARAMS.map((p) => (
                  <Card key={p.key} className="border border-gray-200">
                    <CardHeader className="pb-2">
                      <CardTitle className="text-base flex items-center justify-between">
                        <span className="font-semibold">{p.short}</span>
                        <Badge variant="secondary" className="text-xs">
                          Normal: {p.normal} {p.unit}
                        </Badge>
                      </CardTitle>
                    </CardHeader>
  
                    <CardContent className="space-y-3">
                      <div>
                        <p className="text-sm font-medium text-gray-800">What it is</p>
                        <p className="text-sm text-gray-600">{p.why}</p>
                      </div>
  
                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                        <div className="rounded-lg border border-green-100 bg-green-50 p-3">
                          <p className="text-xs font-semibold text-green-800">When LOW</p>
                          <ul className="text-xs text-green-900 mt-2 list-disc ml-4 space-y-1">
                            {p.low.map((x) => (
                              <li key={x}>{x}</li>
                            ))}
                          </ul>
                        </div>
  
                        <div className="rounded-lg border border-red-100 bg-red-50 p-3">
                          <p className="text-xs font-semibold text-red-800">When HIGH</p>
                          <ul className="text-xs text-red-900 mt-2 list-disc ml-4 space-y-1">
                            {p.high.map((x) => (
                              <li key={x}>{x}</li>
                            ))}
                          </ul>
                        </div>
                      </div>
  
                      <div className="rounded-lg border border-teal-100 bg-teal-50 p-3">
                        <p className="text-xs font-semibold text-teal-900 flex items-center gap-2">
                          <Activity className="w-4 h-4" />
                          Clinical hint
                        </p>
                        <p className="text-xs text-teal-900 mt-2">{p.clinicalHint}</p>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </section>
  
            {/* Pattern Guide */}
            <section>
              <div className="flex items-center gap-2 mb-4">
                <Info className="w-5 h-5 text-gray-700" />
                <h3 className="text-lg font-semibold text-gray-900">Common Liver Injury Patterns (Hints)</h3>
              </div>
  
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {PATTERN_GUIDE.map((item) => (
                  <Card key={item.title} className="border border-gray-200">
                    <CardHeader className="pb-2">
                      <CardTitle className="text-base flex items-center justify-between">
                        {item.title}
                        <Badge variant="outline" className="text-xs">{item.badge}</Badge>
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="text-sm text-gray-600 space-y-3">
                      <p>{item.desc}</p>
                      <p className="text-xs text-gray-700 bg-gray-50 border border-gray-200 p-3 rounded-lg">
                        <span className="font-semibold">Hint: </span>{item.hint}
                      </p>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </section>
  
            {/* Bulk CSV Help */}
            <section>
              <div className="flex items-center gap-2 mb-4">
                <FileSpreadsheet className="w-5 h-5 text-gray-700" />
                <h3 className="text-lg font-semibold text-gray-900">Bulk CSV Upload Guide</h3>
              </div>
  
              <Card className="border border-gray-200">
                <CardContent className="p-5 space-y-4">
                  <p className="text-sm text-gray-700">
                    Your CSV file should contain columns matching the ILPD model input parameters.
                  </p>
  
                  <div className="rounded-lg border border-gray-200 bg-gray-50 p-4">
                    <p className="text-sm font-semibold text-gray-900 mb-2">Expected Columns</p>
                    <code className="text-xs text-gray-700 whitespace-pre-wrap">
                      age, gender, totalBilirubin, directBilirubin, alkalinePhosphatase, sgptAlt, sgotAst, totalProteins, albumin, agRatio
                    </code>
                    <p className="text-xs text-gray-600 mt-3">
                      gender should be Male/Female (or 1/0 depending on your preprocessing).
                    </p>
                  </div>
  
                  <div className="rounded-lg border border-amber-200 bg-amber-50 p-4">
                    <p className="text-sm font-semibold text-amber-900 mb-2">Important</p>
                    <ul className="text-xs text-amber-900 list-disc ml-5 space-y-1">
                      <li>Remove missing values or impute before upload (recommended).</li>
                      <li>Do not include absurd values — the system may reject invalid rows.</li>
                      <li>Keep units consistent with ILPD dataset conventions.</li>
                    </ul>
                  </div>
                </CardContent>
              </Card>
            </section>
  
            {/* FAQ */}
            <section>
              <div className="flex items-center gap-2 mb-4">
                <ChevronDown className="w-5 h-5 text-gray-700" />
                <h3 className="text-lg font-semibold text-gray-900">FAQ</h3>
              </div>
  
              <Accordion type="single" collapsible className="w-full">
                <AccordionItem value="item-1">
                  <AccordionTrigger>Is this a diagnosis?</AccordionTrigger>
                  <AccordionContent>
                    No. This is an ML-based risk prediction support tool. A doctor must confirm diagnosis using clinical evaluation and additional tests.
                  </AccordionContent>
                </AccordionItem>
  
                <AccordionItem value="item-2">
                  <AccordionTrigger>Why does the model show “Low Confidence”?</AccordionTrigger>
                  <AccordionContent>
                    Confidence depends on how clearly your input matches known patterns in training data. Mixed or borderline values often reduce confidence.
                  </AccordionContent>
                </AccordionItem>
  
                <AccordionItem value="item-3">
                  <AccordionTrigger>Can it predict the type of liver disease?</AccordionTrigger>
                  <AccordionContent>
                    Not directly. But patterns (cholestatic/hepatocellular) and contributing factors can suggest possibilities. Exact disease type requires medical evaluation.
                  </AccordionContent>
                </AccordionItem>
  
                <AccordionItem value="item-4">
                  <AccordionTrigger>Why are some inputs highlighted in amber?</AccordionTrigger>
                  <AccordionContent>
                    Amber highlight indicates the value is outside a typical normal range. It’s meant for attention, not diagnosis.
                  </AccordionContent>
                </AccordionItem>
              </Accordion>
            </section>
  
            {/* Navigation buttons */}
            <div className="flex flex-wrap gap-3 pt-2">
              <Button variant="outline" onClick={() => window.scrollTo({ top: 0, behavior: "smooth" })}>
                Back to Top
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }
  