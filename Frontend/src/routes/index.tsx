import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useMemo, useState, type FormEvent } from "react";
import { useQuery } from "@tanstack/react-query";
import { toast } from "sonner";
import {
  AlertTriangle,
  BarChart3,
  Brain,
  Database,
  Gauge,
  Loader2,
  Microscope,
  SlidersHorizontal,
  Sparkles,
} from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  getExamples,
  getFeatureStats,
  getModelInfo,
  getModelSummary,
  postPredict,
  type ExampleCase,
  type FeatureExplanation,
  type FeatureStatsResponse,
  type ModelApproach,
  type ModelInfo,
  type ModelSummary,
  type PredictResponse,
} from "@/lib/api";

const COLORS = {
  benign: "#2563eb",
  malignant: "#dc2626",
  neutral: "#64748b",
};

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Diagnóstico Preditivo - Câncer de Mama (Naive Bayes)" },
      {
        name: "description",
        content:
          "Projeto acadêmico de Machine Learning para classificação de tumores benignos e malignos usando Naive Bayes e o dataset WDBC.",
      },
    ],
  }),
  component: Index,
});

function Index() {
  const modelInfoQuery = useQuery({
    queryKey: ["model-info"],
    queryFn: getModelInfo,
    retry: 1,
  });
  const approaches = modelInfoQuery.data?.approaches ?? [];
  const defaultApproachId = modelInfoQuery.data?.default_approach_id ?? approaches[0]?.id ?? "";
  const [selectedApproachId, setSelectedApproachId] = useState("");
  const effectiveApproachId = selectedApproachId || defaultApproachId;
  const selectedApproach =
    approaches.find((approach) => approach.id === effectiveApproachId) ?? approaches[0] ?? null;

  const examplesQuery = useQuery({
    queryKey: ["examples", effectiveApproachId],
    queryFn: () => getExamples(effectiveApproachId),
    retry: 1,
    enabled: Boolean(effectiveApproachId),
  });
  const summaryQuery = useQuery({
    queryKey: ["model-summary"],
    queryFn: getModelSummary,
    retry: 1,
  });
  const featureStatsQuery = useQuery({
    queryKey: ["feature-stats"],
    queryFn: getFeatureStats,
    retry: 1,
  });

  const features = useMemo<string[]>(() => {
    const modelFeatures = selectedApproach?.features ?? modelInfoQuery.data?.features;
    return Array.isArray(modelFeatures) ? modelFeatures : [];
  }, [modelInfoQuery.data, selectedApproach]);

  const examples = useMemo(() => examplesQuery.data?.examples ?? [], [examplesQuery.data]);
  const [selectedExampleId, setSelectedExampleId] = useState("");
  const [values, setValues] = useState<Record<string, string>>({});
  const [submitting, setSubmitting] = useState(false);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [mode, setMode] = useState<"simple" | "advanced">("simple");
  const [threshold, setThreshold] = useState(0.5);

  useEffect(() => {
    if (!features.length) return;
    setValues((previous) => {
      const next: Record<string, string> = {};
      for (const feature of features) next[feature] = previous[feature] ?? "";
      return next;
    });
  }, [features]);

  useEffect(() => {
    setSelectedExampleId("");
    setResult(null);
  }, [effectiveApproachId]);

  useEffect(() => {
    if (!selectedExampleId && examples.length) {
      applyExample(examples[0]);
    }
  }, [examples, selectedExampleId]);

  const selectedExample = examples.find((example) => example.id === selectedExampleId) ?? null;

  const applyExample = (example: ExampleCase) => {
    setSelectedExampleId(example.id);
    setResult(null);
    setValues(
      Object.fromEntries(
        Object.entries(example.features).map(([feature, value]) => [feature, String(value)]),
      ),
    );
  };

  const handleChange = (name: string, value: string) => {
    setValues((previous) => ({ ...previous, [name]: value }));
  };

  const buildPayload = () => {
    const payload: Record<string, number> = {};
    for (const feature of features) {
      const raw = values[feature]?.trim();
      if (!raw) {
        throw new Error("Preencha todos os campos antes de enviar.");
      }
      const numericValue = Number(raw);
      if (!Number.isFinite(numericValue)) {
        throw new Error(`O campo "${feature}" precisa ser um número válido.`);
      }
      payload[feature] = numericValue;
    }
    return payload;
  };

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault();

    let payload: Record<string, number>;
    try {
      payload = buildPayload();
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Dados inválidos.");
      return;
    }

    setSubmitting(true);
    setResult(null);
    try {
      const response = await postPredict(payload, threshold, effectiveApproachId);
      setResult(response);
      toast.success("Predição gerada com sucesso.");
    } catch (error) {
      const message = error instanceof Error ? error.message : "Falha ao chamar /predict";
      toast.error(message);
    } finally {
      setSubmitting(false);
    }
  };

  const loadingInitial =
    modelInfoQuery.isLoading ||
    examplesQuery.isLoading ||
    summaryQuery.isLoading ||
    featureStatsQuery.isLoading;
  const hasInitialError =
    modelInfoQuery.isError ||
    examplesQuery.isError ||
    summaryQuery.isError ||
    featureStatsQuery.isError;

  return (
    <div className="min-h-screen bg-background text-foreground">
      <Hero />

      <main className="mx-auto w-full max-w-7xl px-4 pb-24 pt-10 sm:px-6 lg:px-8">
        <Alert className="mb-8 border-warning/40 bg-warning/10 text-warning-foreground">
          <AlertTriangle className="size-4" />
          <AlertTitle>Aviso acadêmico</AlertTitle>
          <AlertDescription>
            Este é um projeto acadêmico e <strong>não</strong> uma ferramenta médica. A predição
            ajuda a explicar o fluxo de Machine Learning, mas não substitui avaliação clínica.
          </AlertDescription>
        </Alert>

        {loadingInitial && (
          <Card>
            <CardContent className="flex items-center gap-2 py-10 text-muted-foreground">
              <Loader2 className="size-4 animate-spin" />
              Carregando modelo, exemplos e estatísticas...
            </CardContent>
          </Card>
        )}

        {hasInitialError && (
          <Alert variant="destructive" className="mb-8">
            <AlertTitle>Não foi possível carregar todos os dados</AlertTitle>
            <AlertDescription>
              Verifique se o backend Flask está rodando em <code>http://localhost:5000</code>.
            </AlertDescription>
          </Alert>
        )}

        {!loadingInitial && !hasInitialError && (
          <div className="space-y-8">
            <section className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_390px]">
              <PredictionPanel
                examples={examples}
                selectedExample={selectedExample}
                approaches={approaches}
                selectedApproach={selectedApproach}
                features={features}
                values={values}
                mode={mode}
                threshold={threshold}
                submitting={submitting}
                onModeChange={setMode}
                onApproachChange={setSelectedApproachId}
                onThresholdChange={setThreshold}
                onExampleChange={(id) => {
                  const example = examples.find((item) => item.id === id);
                  if (example) applyExample(example);
                }}
                onValueChange={handleChange}
                onSubmit={handleSubmit}
              />

              <aside className="space-y-6">
                <ResultCard result={result} loading={submitting} />
                <MetaCard modelInfo={modelInfoQuery.data ?? null} />
              </aside>
            </section>

            {result && (
              <section className="grid gap-6 lg:grid-cols-2">
                <ProbabilityCard result={result} />
                <FeatureExplanationCard result={result} stats={featureStatsQuery.data ?? null} />
              </section>
            )}

            <section className="grid gap-6 lg:grid-cols-[1fr_1fr]">
              <ModelPerformanceCard summary={summaryQuery.data ?? null} />
              <ApproachComparisonCard
                summary={summaryQuery.data ?? null}
                selectedApproachId={effectiveApproachId}
              />
              <DatasetCard summary={summaryQuery.data ?? null} examples={examples} />
            </section>
          </div>
        )}
      </main>
    </div>
  );
}

function Hero() {
  return (
    <header
      className="relative overflow-hidden text-primary-foreground"
      style={{ background: "var(--gradient-hero)" }}
    >
      <div className="relative mx-auto max-w-7xl px-4 py-16 sm:px-6 sm:py-20 lg:px-8">
        <Badge className="mb-5 border-white/30 bg-white/15 text-primary-foreground backdrop-blur">
          Projeto Acadêmico de Machine Learning
        </Badge>
        <h1 className="text-4xl font-bold tracking-tight sm:text-5xl">
          Diagnóstico Preditivo de Câncer de Mama
        </h1>
        <p className="mt-4 max-w-3xl text-base text-primary-foreground/85 sm:text-lg">
          Interface para testar um modelo Naive Bayes treinado com a base WDBC, visualizar
          confiança, comparar variáveis com padrões da base e entender onde o modelo pode ficar
          incerto.
        </p>
        <div className="mt-8 flex flex-wrap gap-3 text-sm">
          <span className="inline-flex items-center gap-2 rounded-full bg-white/15 px-3 py-1.5 backdrop-blur">
            <Brain className="size-4" />
            Algoritmo: Naive Bayes
          </span>
          <span className="inline-flex items-center gap-2 rounded-full bg-white/15 px-3 py-1.5 backdrop-blur">
            <Database className="size-4" />
            Base: Breast Cancer Wisconsin Diagnostic Dataset (UCI)
          </span>
        </div>
      </div>
    </header>
  );
}

function PredictionPanel({
  examples,
  selectedExample,
  approaches,
  selectedApproach,
  features,
  values,
  mode,
  threshold,
  submitting,
  onModeChange,
  onApproachChange,
  onThresholdChange,
  onExampleChange,
  onValueChange,
  onSubmit,
}: {
  examples: ExampleCase[];
  selectedExample: ExampleCase | null;
  approaches: ModelApproach[];
  selectedApproach: ModelApproach | null;
  features: string[];
  values: Record<string, string>;
  mode: "simple" | "advanced";
  threshold: number;
  submitting: boolean;
  onModeChange: (mode: "simple" | "advanced") => void;
  onApproachChange: (id: string) => void;
  onThresholdChange: (threshold: number) => void;
  onExampleChange: (id: string) => void;
  onValueChange: (name: string, value: string) => void;
  onSubmit: (event: FormEvent) => void;
}) {
  return (
    <Card className="shadow-sm">
      <CardHeader className="space-y-4">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <CardTitle className="flex items-center gap-2 text-xl">
              <Microscope className="size-5" />
              Predição
            </CardTitle>
            <p className="mt-1 text-sm text-muted-foreground">
              Use um exemplo real da base ou ajuste manualmente as features no modo avançado.
            </p>
          </div>
          <div className="inline-flex rounded-md border bg-muted p-1">
            <Button
              type="button"
              size="sm"
              variant={mode === "simple" ? "default" : "ghost"}
              onClick={() => onModeChange("simple")}
            >
              Simples
            </Button>
            <Button
              type="button"
              size="sm"
              variant={mode === "advanced" ? "default" : "ghost"}
              onClick={() => onModeChange("advanced")}
            >
              Avançado
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <form onSubmit={onSubmit} className="space-y-6">
          <div className="grid gap-4 lg:grid-cols-[1fr_1fr_260px]">
            <div className="space-y-2">
              <Label htmlFor="approach">Abordagem Naive Bayes</Label>
              <select
                id="approach"
                value={selectedApproach?.id ?? ""}
                onChange={(event) => onApproachChange(event.target.value)}
                className="h-10 w-full rounded-md border border-input bg-background px-3 text-sm"
              >
                {approaches.map((approach) => (
                  <option key={approach.id} value={approach.id}>
                    {approach.name}
                  </option>
                ))}
              </select>
              {selectedApproach && (
                <p className="text-xs text-muted-foreground">
                  {selectedApproach.features_count} features · F1 macro{" "}
                  {formatPercent(selectedApproach.metrics.f1_macro)} · recall maligno{" "}
                  {formatPercent(selectedApproach.metrics.recall_malignant)}
                </p>
              )}
            </div>

            <div className="space-y-2">
              <Label htmlFor="example">Exemplo pronto</Label>
              <select
                id="example"
                value={selectedExample?.id ?? ""}
                onChange={(event) => onExampleChange(event.target.value)}
                className="h-10 w-full rounded-md border border-input bg-background px-3 text-sm"
              >
                {examples.map((example) => (
                  <option key={example.id} value={example.id}>
                    {example.name} - {example.actual_label}
                  </option>
                ))}
              </select>
              {selectedExample && (
                <p className="text-xs text-muted-foreground">
                  Classe real: <strong>{selectedExample.actual_label}</strong>. Confiança original
                  do modelo nesta amostra: {formatPercent(selectedExample.confidence)}.
                </p>
              )}
            </div>

            <div className="space-y-2">
              <Label htmlFor="threshold" className="flex items-center gap-2">
                <SlidersHorizontal className="size-4" />
                Limiar para maligno: {formatPercent(threshold)}
              </Label>
              <Input
                id="threshold"
                type="range"
                min="0.1"
                max="0.9"
                step="0.05"
                value={threshold}
                onChange={(event) => onThresholdChange(Number(event.target.value))}
              />
              <p className="text-xs text-muted-foreground">
                Menor limiar aumenta sensibilidade para maligno; maior limiar exige mais certeza.
              </p>
            </div>
          </div>

          {mode === "advanced" && (
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-3">
              {features.map((feature) => (
                <div key={feature} className="space-y-1.5">
                  <Label htmlFor={feature} className="text-xs font-medium">
                    {feature}
                  </Label>
                  <Input
                    id={feature}
                    name={feature}
                    type="number"
                    step="any"
                    inputMode="decimal"
                    required
                    value={values[feature] ?? ""}
                    onChange={(event) => onValueChange(feature, event.target.value)}
                    placeholder="0.00"
                  />
                </div>
              ))}
            </div>
          )}

          {mode === "simple" && selectedExample && (
            <div className="grid gap-3 rounded-md border bg-muted/30 p-4 text-sm sm:grid-cols-2 lg:grid-cols-3">
              {features.slice(0, 6).map((feature) => (
                <div key={feature}>
                  <span className="text-muted-foreground">{feature}</span>
                  <p className="font-medium">{formatNumber(selectedExample.features[feature])}</p>
                </div>
              ))}
            </div>
          )}

          <div className="flex flex-wrap items-center justify-between gap-3 border-t pt-4">
            <p className="text-xs text-muted-foreground">
              {features.length} features carregadas de <code>GET /model-info</code>
            </p>
            <Button type="submit" size="lg" disabled={submitting || !features.length}>
              {submitting ? (
                <>
                  <Loader2 className="size-4 animate-spin" />
                  Processando...
                </>
              ) : (
                <>
                  <Sparkles className="size-4" />
                  Prever
                </>
              )}
            </Button>
          </div>
        </form>
      </CardContent>
    </Card>
  );
}

function ResultCard({ result, loading }: { result: PredictResponse | null; loading: boolean }) {
  const confidence = result?.confidence ?? null;
  const malignant = result?.class === "M";
  const tone =
    confidence === null
      ? ""
      : malignant
        ? "border-destructive/40 bg-destructive/5"
        : "border-success/40 bg-success/5";

  return (
    <Card className={tone}>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base">
          <Gauge className="size-4" />
          Resultado
        </CardTitle>
      </CardHeader>
      <CardContent>
        {loading && (
          <div className="flex items-center gap-2 py-6 text-sm text-muted-foreground">
            <Loader2 className="size-4 animate-spin" />
            Processando predição...
          </div>
        )}

        {!loading && !result && (
          <p className="py-6 text-sm text-muted-foreground">
            Escolha um exemplo e clique em <strong>Prever</strong> para ver a classificação.
          </p>
        )}

        {!loading && result && (
          <div className="space-y-4">
            <div>
              <p className="text-xs uppercase tracking-wide text-muted-foreground">Classificação</p>
              <p
                className={`mt-1 text-2xl font-semibold ${malignant ? "text-destructive" : "text-success"}`}
              >
                {result.prediction}
              </p>
            </div>
            <div className="grid grid-cols-2 gap-3">
              <Metric label="Confiança" value={formatPercent(confidence ?? 0)} />
              <Metric label="Nível" value={String(result.confidence_level ?? "-")} />
            </div>
            {confidence !== null && confidence < 0.7 && (
              <Alert className="border-warning/40 bg-warning/10">
                <AlertTriangle className="size-4" />
                <AlertTitle>Caso incerto</AlertTitle>
                <AlertDescription>
                  O modelo está abaixo de 70% de confiança. Esse resultado merece cautela.
                </AlertDescription>
              </Alert>
            )}
            <p className="text-xs text-muted-foreground">
              Limiar de maligno usado: {formatPercent(result.threshold ?? 0.5)}.
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function ProbabilityCard({ result }: { result: PredictResponse }) {
  const data = Object.entries(result.probabilities ?? {}).map(([label, value]) => ({
    label,
    value,
    percent: Number((value * 100).toFixed(2)),
  }));

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base">
          <BarChart3 className="size-4" />
          Probabilidades do modelo
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-5">
        <div className="h-56">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data} layout="vertical" margin={{ left: 18, right: 20 }}>
              <CartesianGrid strokeDasharray="3 3" horizontal={false} />
              <XAxis type="number" domain={[0, 100]} tickFormatter={(value) => `${value}%`} />
              <YAxis type="category" dataKey="label" width={110} />
              <Tooltip formatter={(value) => `${Number(value).toFixed(1)}%`} />
              <Bar dataKey="percent" radius={[0, 6, 6, 0]}>
                {data.map((item) => (
                  <Cell
                    key={item.label}
                    fill={item.label.includes("Maligno") ? COLORS.malignant : COLORS.benign}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
        <p className="text-sm text-muted-foreground">{result.explanation}</p>
      </CardContent>
    </Card>
  );
}

function FeatureExplanationCard({
  result,
  stats,
}: {
  result: PredictResponse;
  stats: FeatureStatsResponse | null;
}) {
  const topFeatures = result.top_features ?? [];
  const chartData = topFeatures.slice(0, 5).flatMap((feature) => [
    { feature: feature.feature, grupo: "Amostra", value: feature.value },
    { feature: feature.feature, grupo: "Média benigna", value: feature.benign_mean },
    { feature: feature.feature, grupo: "Média maligna", value: feature.malignant_mean },
  ]);

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Features que mais se destacaram</CardTitle>
      </CardHeader>
      <CardContent className="space-y-5">
        <div className="h-72">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="feature" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip formatter={(value) => formatNumber(Number(value))} />
              <Legend />
              <Bar dataKey="value" name="Valor" radius={[4, 4, 0, 0]}>
                {chartData.map((item, index) => (
                  <Cell
                    key={`${item.feature}-${item.grupo}-${index}`}
                    fill={
                      item.grupo === "Amostra"
                        ? COLORS.neutral
                        : item.grupo.includes("maligna")
                          ? COLORS.malignant
                          : COLORS.benign
                    }
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="space-y-3">
          {topFeatures.map((feature) => (
            <FeatureRow key={feature.feature} feature={feature} />
          ))}
        </div>

        {stats && (
          <p className="text-xs text-muted-foreground">
            Comparação calculada com médias e desvios da base WDBC carregada em{" "}
            <code>GET /feature-stats</code>.
          </p>
        )}
      </CardContent>
    </Card>
  );
}

function FeatureRow({ feature }: { feature: FeatureExplanation }) {
  return (
    <div className="rounded-md border p-3 text-sm">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <strong>{feature.feature}</strong>
        <Badge variant={feature.closer_to === "M" ? "destructive" : "default"}>
          Mais próximo de {feature.closer_to_label}
        </Badge>
      </div>
      <div className="mt-2 grid gap-2 text-xs text-muted-foreground sm:grid-cols-3">
        <span>Amostra: {formatNumber(feature.value)}</span>
        <span>Média benigna: {formatNumber(feature.benign_mean)}</span>
        <span>Média maligna: {formatNumber(feature.malignant_mean)}</span>
      </div>
    </div>
  );
}

function ModelPerformanceCard({ summary }: { summary: ModelSummary | null }) {
  const metrics = summary?.metrics ?? {};
  const confusion = parseConfusionMatrix(summary);

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Desempenho do modelo</CardTitle>
      </CardHeader>
      <CardContent className="space-y-5">
        <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
          <Metric label="Acurácia" value={formatPercent(metrics.accuracy)} />
          <Metric label="Precisão macro" value={formatPercent(metrics.precision_macro)} />
          <Metric label="Recall macro" value={formatPercent(metrics.recall_macro)} />
          <Metric label="F1 macro" value={formatPercent(metrics.f1_macro)} />
        </div>
        <div className="rounded-md border p-4">
          <p className="mb-3 text-sm font-medium">Matriz de confusão</p>
          <div className="grid grid-cols-3 gap-2 text-center text-sm">
            <div />
            <div className="rounded bg-muted p-2">Prev. Benigno</div>
            <div className="rounded bg-muted p-2">Prev. Maligno</div>
            <div className="rounded bg-muted p-2">Real Benigno</div>
            <div className="rounded border p-2 font-semibold">{confusion.bb}</div>
            <div className="rounded border p-2 font-semibold">{confusion.bm}</div>
            <div className="rounded bg-muted p-2">Real Maligno</div>
            <div className="rounded border p-2 font-semibold">{confusion.mb}</div>
            <div className="rounded border p-2 font-semibold">{confusion.mm}</div>
          </div>
        </div>
        <p className="text-sm text-muted-foreground">
          Baseline ingênuo de classe majoritária:{" "}
          <strong>{formatPercent(summary?.majority_baseline_accuracy)}</strong>. O modelo deve ser
          comparado contra esse valor para mostrar que aprendeu além de sempre prever benigno.
        </p>
      </CardContent>
    </Card>
  );
}

function ApproachComparisonCard({
  summary,
  selectedApproachId,
}: {
  summary: ModelSummary | null;
  selectedApproachId: string;
}) {
  const experiments = summary?.experiments ?? [];
  const chartData = experiments.map((experiment) => ({
    name: experiment.name.replace(" - ", "\n"),
    id: experiment.id,
    f1: Number(((experiment.f1_macro ?? 0) * 100).toFixed(2)),
    recallMaligno: Number(((experiment.recall_malignant ?? 0) * 100).toFixed(2)),
    confiancaErro: Number(((experiment.mean_error_confidence ?? 0) * 100).toFixed(2)),
  }));
  const selected = experiments.find((experiment) => experiment.id === selectedApproachId);

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Comparação científica das abordagens</CardTitle>
      </CardHeader>
      <CardContent className="space-y-5">
        <div className="h-72">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" tick={{ fontSize: 10 }} interval={0} />
              <YAxis tickFormatter={(value) => `${value}%`} />
              <Tooltip formatter={(value) => `${Number(value).toFixed(1)}%`} />
              <Legend />
              <Bar dataKey="f1" name="F1 macro" fill="#2563eb" radius={[4, 4, 0, 0]} />
              <Bar
                dataKey="recallMaligno"
                name="Recall maligno"
                fill="#dc2626"
                radius={[4, 4, 0, 0]}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {selected && (
          <div className="rounded-md border p-4 text-sm">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <strong>{selected.name}</strong>
              {summary?.default_approach_id === selected.id && <Badge>Recomendada</Badge>}
            </div>
            <p className="mt-2 text-muted-foreground">{selected.description}</p>
            <div className="mt-3 grid gap-2 sm:grid-cols-3">
              <Metric label="Features" value={String(selected.features_count)} />
              <Metric label="Brier maligno" value={formatNumber(selected.brier_malignant)} />
              <Metric
                label="Conf. média nos erros"
                value={formatPercent(selected.mean_error_confidence)}
              />
            </div>
          </div>
        )}

        <p className="text-xs text-muted-foreground">
          Todos os experimentos mantêm Gaussian Naive Bayes. A comparação muda apenas seleção de
          features, redundância e calibração de probabilidades.
        </p>
      </CardContent>
    </Card>
  );
}

function DatasetCard({
  summary,
  examples,
}: {
  summary: ModelSummary | null;
  examples: ExampleCase[];
}) {
  const distribution = summary?.dataset.class_distribution;
  const pieData = distribution
    ? Object.entries(distribution).map(([key, value]) => ({
        name: value.label,
        value: value.count,
        key,
      }))
    : [];

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Base e exemplos disponíveis</CardTitle>
      </CardHeader>
      <CardContent className="space-y-5">
        <div className="grid grid-cols-2 gap-3">
          <Metric label="Observações" value={String(summary?.dataset.observations ?? "-")} />
          <Metric label="Features" value={String(summary?.dataset.features ?? "-")} />
        </div>
        <div className="h-52">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie data={pieData} dataKey="value" nameKey="name" outerRadius={72} label>
                {pieData.map((item) => (
                  <Cell key={item.key} fill={item.key === "M" ? COLORS.malignant : COLORS.benign} />
                ))}
              </Pie>
              <Tooltip />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </div>
        <div className="space-y-2">
          {examples.map((example) => (
            <div
              key={example.id}
              className="flex items-center justify-between rounded-md border px-3 py-2 text-sm"
            >
              <span>{example.name}</span>
              <span className="text-muted-foreground">
                {example.actual_label} · conf. {formatPercent(example.confidence)}
              </span>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

function MetaCard({ modelInfo }: { modelInfo: ModelInfo | null }) {
  const classes = modelInfo?.classes
    ? Object.values(modelInfo.classes).join(" / ")
    : "Tumor Benigno / Tumor Maligno";

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Sobre o modelo</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3 text-sm">
        <Row label="Algoritmo" value={modelInfo?.model ?? "GaussianNB"} />
        <Row label="Alvo" value={modelInfo?.target ?? "Diagnosis"} />
        <Row label="Features" value={modelInfo?.features?.length?.toString() ?? "-"} />
        <Row label="Classes" value={classes} />
      </CardContent>
    </Card>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between gap-3 border-b border-border/60 pb-2 last:border-b-0 last:pb-0">
      <span className="text-muted-foreground">{label}</span>
      <span className="text-right font-medium text-foreground">{value}</span>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border bg-background p-3">
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className="mt-1 text-lg font-semibold">{value}</p>
    </div>
  );
}

function parseConfusionMatrix(summary: ModelSummary | null) {
  const matrix = summary?.metrics.confusion_matrix;
  return {
    bb: matrix?.["previsto_Tumor Benigno"]?.["real_Tumor Benigno"] ?? "-",
    bm: matrix?.["previsto_Tumor Maligno"]?.["real_Tumor Benigno"] ?? "-",
    mb: matrix?.["previsto_Tumor Benigno"]?.["real_Tumor Maligno"] ?? "-",
    mm: matrix?.["previsto_Tumor Maligno"]?.["real_Tumor Maligno"] ?? "-",
  };
}

function formatPercent(value: number | undefined | null) {
  if (typeof value !== "number" || !Number.isFinite(value)) return "-";
  return `${(value * 100).toFixed(1)}%`;
}

function formatNumber(value: number | undefined | null) {
  if (typeof value !== "number" || !Number.isFinite(value)) return "-";
  if (Math.abs(value) >= 100) return value.toFixed(1);
  if (Math.abs(value) >= 1) return value.toFixed(3);
  return value.toFixed(5);
}
