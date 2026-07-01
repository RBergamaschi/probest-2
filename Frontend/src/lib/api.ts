const RAW_BASE =
  (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? "http://localhost:5000";

export const API_BASE_URL = RAW_BASE.replace(/\/$/, "");

export interface ModelInfo {
  features: string[];
  all_features?: string[];
  classes?: Record<string, string>;
  model?: string;
  target?: string;
  default_approach_id?: string;
  approaches?: ModelApproach[];
  input_format?: {
    features?: Record<string, "number" | string>;
  };
  [key: string]: unknown;
}

export interface ModelApproach {
  id: string;
  name: string;
  description: string;
  features: string[];
  features_count: number;
  metrics: ApproachMetrics;
}

export interface ApproachMetrics {
  accuracy?: number;
  precision_macro?: number;
  recall_macro?: number;
  f1_macro?: number;
  recall_malignant?: number;
  precision_malignant?: number;
  brier_malignant?: number;
  mean_error_confidence?: number;
  [key: string]: unknown;
}

export interface PredictResponse {
  prediction: string | number;
  class?: string;
  label?: string;
  threshold?: number;
  confidence?: number;
  confidence_level?: "alta" | "moderada" | "baixa" | string;
  explanation?: string;
  probability?: number;
  probabilities?: Record<string, number>;
  raw_probabilities?: Record<string, number>;
  top_features?: FeatureExplanation[];
  approach_id?: string;
  approach_name?: string;
  features_used?: string[];
  features_count?: number;
  [key: string]: unknown;
}

export interface FeatureExplanation {
  feature: string;
  value: number;
  z_score: number;
  abs_z_score: number;
  closer_to: "B" | "M";
  closer_to_label: string;
  distance_to_benign_mean: number;
  distance_to_malignant_mean: number;
  benign_mean: number;
  malignant_mean: number;
  overall_mean: number;
}

export interface ExampleCase {
  id: string;
  name: string;
  actual_class: "B" | "M";
  actual_label: string;
  model_prediction: "B" | "M";
  probability_malignant: number;
  confidence: number;
  features: Record<string, number>;
}

export interface ExamplesResponse {
  examples: ExampleCase[];
}

export interface FeatureStatsResponse {
  features: Record<
    string,
    {
      overall: { mean: number; std: number; min: number; max: number };
      classes: Record<
        "B" | "M",
        {
          label: string;
          mean: number;
          std: number;
        }
      >;
    }
  >;
}

export interface ModelSummary {
  dataset: {
    name: string;
    observations: number;
    features: number;
    class_distribution: Record<"B" | "M", { label: string; count: number; percentage: number }>;
  };
  majority_baseline_accuracy: number;
  metrics: {
    accuracy?: number;
    precision_macro?: number;
    recall_macro?: number;
    f1_macro?: number;
    confusion_matrix?: Record<string, Record<string, number>>;
    classification_report?: Record<string, Record<string, number> | number>;
    [key: string]: unknown;
  };
  experiments?: Array<
    ApproachMetrics & {
      id: string;
      name: string;
      description: string;
      features_count: number;
      features: string[];
    }
  >;
  approaches?: ModelApproach[];
  default_approach_id?: string;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    let detail = "";
    try {
      const data = await res.json();
      detail =
        (data as { error?: string; message?: string }).error ??
        (data as { message?: string }).message ??
        "";
    } catch {
      detail = await res.text().catch(() => "");
    }
    throw new Error(`Erro ${res.status} ao chamar ${path}${detail ? `: ${detail}` : ""}`);
  }
  return (await res.json()) as T;
}

export const getModelInfo = () => request<ModelInfo>("/model-info");

export const postPredict = (
  features: Record<string, number>,
  threshold = 0.5,
  approachId?: string,
) =>
  request<PredictResponse>("/predict", {
    method: "POST",
    body: JSON.stringify({ features, threshold, approach_id: approachId }),
  });

export const getHealth = () => request<{ status: string }>("/health");

export const getExamples = (approachId?: string) =>
  request<ExamplesResponse>(`/examples${approachId ? `?approach_id=${approachId}` : ""}`);

export const getFeatureStats = () => request<FeatureStatsResponse>("/feature-stats");

export const getModelSummary = () => request<ModelSummary>("/model-summary");
