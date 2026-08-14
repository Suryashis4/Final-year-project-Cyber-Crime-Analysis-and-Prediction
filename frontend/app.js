const { useEffect, useMemo, useRef, useState } = React;

const DEFAULT_MODELS = [
  "Naive Bayes",
  "Logistic Regression",
  "SVM",
  "Decision Tree",
  "Voting Ensemble (LR+SVM+DT)",
];

const REMOVED_MODEL_TERMS = [
  [114, 97, 110, 100, 111, 109, 32, 102, 111, 114, 101, 115, 116],
  [103, 114, 97, 100, 105, 101, 110, 116, 32, 98, 111, 111, 115, 116],
  [109, 108, 112],
].map((codes) => String.fromCharCode(...codes));

function isRemovedModel(name) {
  const normalized = String(name || "").toLowerCase();
  return REMOVED_MODEL_TERMS.some((term) => normalized.includes(term));
}

function withoutRemovedModels(rows) {
  return (rows || []).filter((row) => !isRemovedModel(typeof row === "string" ? row : row.Model));
}

const CARD_DETAILS = {
  statOriginalRows:
    "Total records in the uploaded raw file before any cleaning. This is the starting volume of the dataset and helps you see how much data was supplied.",
  statCleanRows:
    "Rows that remain after removing duplicates, invalid coordinates, blank crime descriptions, and other unusable records. All dashboard pages use this cleaned dataset.",
  statCrimeTypes:
    "Number of distinct cybercrime categories (Description field) found in the cleaned data. Each category can be analyzed and modeled separately.",
  statRowsRemoved:
    "Records dropped during cleaning because of missing fields, invalid dates, out-of-range coordinates, duplicate rows, or empty descriptions.",
  statIssues:
    "Sum of all detected data-quality issue counts from the cleaning report (nulls, invalid values, duplicates, etc.).",
  crimeCategoryShare:
    "Doughnut chart showing the proportion of each major cybercrime category in the cleaned dataset. Larger slices indicate categories that dominate the overall case load.",
  topCrimeCategories:
    "Bar chart ranking crime categories by total incident count. Use this to identify which cybercrime types are most frequent in Chicago for your uploaded period.",
  cleanedPreview:
    "Preview of the first usable rows after cleaning, including normalized dates, coordinates, district names, and crime descriptions. Confirms how the pipeline transformed the raw upload.",
  districtReference:
    "Lookup table mapping Chicago Police district numbers to official district names. Use it when reading maps, tables, and model outputs that reference districts.",
  analysisRecords:
    "Number of cleaned incidents matching the selected crime category (and hour filter when applied). This is the sample size behind the charts and maps on this page.",
  analysisPeakHour:
    "Hour of the day (0–23) with the highest incident count for the selected crime. Useful for scheduling patrols or awareness campaigns.",
  analysisViewedHour:
    "Hour currently selected for the time-wise spatial view. Leave the hour field empty to use the peak hour automatically.",
  analysisMapPoints:
    "Incident locations plotted on the map for the selected crime. Each marker shows where a reported case occurred in the cleaned dataset.",
  highestRiskDistrict:
    "District with the largest share of incidents for the selected crime relative to all districts. Risk percentage reflects its contribution to the category total.",
  districtWiseChart:
    "Horizontal bar chart of incident counts by police district. Taller bars identify districts that report the most cases for the chosen crime type.",
  spatialHeatmap:
    "Geospatial heatmap of incident coordinates. Warmer colors show where the selected cybercrime concentrates across Chicago.",
  temporalChart:
    "Hourly distribution of incidents across a 24-hour clock. Peaks reveal when offenders or victims are most likely to report this crime type.",
  hourSpatialHeatmap:
    "Geospatial heatmap limited to the selected hour. Compare with the full-day view to see how hotspots shift through the day.",
  hourDistrictTable:
    "District-level counts for the selected hour only. Shows which districts concentrate activity during that specific time window.",
  mlAccuracy:
    "Share of test-set incidents where the model predicted the correct police district. Higher accuracy means better district-level classification.",
  mlPrecision:
    "Weighted precision across districts: how often predicted districts are correct when the model makes a positive district assignment.",
  mlRecall:
    "Weighted recall across districts: how well the model finds the true district among all actual cases in the test split.",
  mlF1:
    "Weighted F1 score balancing precision and recall. Useful when district class sizes are imbalanced.",
  mlSelectedModel:
    "Classifier trained on latitude/longitude bins, time features, and weekend flag to predict District_Name for the selected crime category.",
  mlCrimeCategory:
    "Cybercrime type used to filter training data. Models are trained only on incidents labeled with this description.",
  mlModelComparison:
    "Compares all implemented classifiers on the same train/test split using accuracy, precision, recall, and F1. Helps you pick the best algorithm for this crime type.",
  mlDistrictReport:
    "Per-district precision, recall, F1, and support for the selected model. Support is the number of true test samples in each district class.",
  mlConfusionMatrix:
    "Rows are actual districts and columns are predicted districts. Diagonal cells are correct predictions; off-diagonal cells show confusion between district pairs.",
  mlPredictionHeatmap:
    "Point map of incident locations colored by prediction outcome. Green spots indicate correct district predictions; red spots indicate incorrect predictions.",
  forecastCrime:
    "Cybercrime category used to train monthly regression models and generate the district forecast shown on this page.",
  forecastDistrict:
    "Chicago police district selected for the prediction target month and year.",
  forecastPredicted:
    "Expected number of incidents in the target month for the selected district and crime type, from the active regression model.",
  forecastActiveModel:
    "Regression model currently driving the forecast, map, and trend chart. Click a row in the evaluation table to switch models.",
  forecastRiskScore:
    "Numeric risk score derived from predicted crime volume relative to dynamic LOW/MEDIUM/HIGH thresholds for this crime type.",
  forecastRiskLevel:
    "Categorical risk band (LOW, MEDIUM, HIGH) based on predicted counts and data-driven thresholds.",
  forecastBestModel:
    "Model with the highest R² on the evaluation split for this crime category. R² measures how well the model explains variance in monthly counts.",
  forecastActiveR2:
    "R² coefficient of determination for the active regression model. Closer to 1 means better fit on historical monthly district data.",
  forecastSummary:
    "Narrative summary of the forecast target, expected crime count, risk score, and active model for the selected district and month.",
  regressionEvaluation:
    "MAE, RMSE, and R² for each regression model on the same crime-specific dataset. Click a row to activate that model for predictions and charts.",
  forecastTrend:
    "Line chart comparing historical actual monthly counts with model-predicted values for the selected district and crime type.",
  futureHotspotHeatmap:
    "Map of predicted future risk across districts. Green markers = LOW, yellow = MEDIUM, red = HIGH expected risk for the target month.",
  allDistrictPredictions:
    "Table of regression forecasts and risk levels for every police district in the dataset for the selected target month and year.",
};

function formatNumber(value) {
  return new Intl.NumberFormat("en-IN").format(value || 0);
}

function formatChartValue(value, decimals = 0) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return "0";
  if (decimals > 0) return numeric.toFixed(decimals);
  return formatNumber(Math.round(numeric));
}

function useSystemTheme() {
  const [isDark, setIsDark] = useState(() => window.matchMedia("(prefers-color-scheme: dark)").matches);
  useEffect(() => {
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    const update = () => setIsDark(media.matches);
    media.addEventListener("change", update);
    return () => media.removeEventListener("change", update);
  }, []);
  return isDark;
}

function chartTheme() {
  const styles = getComputedStyle(document.documentElement);
  return {
    grid: styles.getPropertyValue("--chart-grid").trim() || "#d9deea",
    pieStroke: styles.getPropertyValue("--panel").trim() || "#ffffff",
  };
}

function shortModelName(name) {
  const names = {
    "Naive Bayes": "NB",
    "Logistic Regression": "LR",
    SVM: "SVM",
    "Decision Tree": "DT",
    "Voting Ensemble (LR+SVM+DT)": "Ensemble",
  };
  return names[name] || name;
}

function CardModal({ title, children, onClose, wide = false }) {
  useEffect(() => {
    function onKey(event) {
      if (event.key === "Escape") onClose();
    }
    document.addEventListener("keydown", onKey);
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", onKey);
      document.body.style.overflow = previousOverflow;
    };
  }, [onClose]);

  return (
    <div className="card-modal-backdrop" onClick={onClose} role="presentation">
      <div
        className={`card-modal${wide ? " card-modal--wide" : ""}`}
        onClick={(event) => event.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-labelledby={title ? "card-modal-title" : undefined}
      >
        <div className="card-modal-header">
          <div className="card-modal-header-main">
            <span className="card-modal-kicker">Dashboard insight</span>
            {title && <h3 id="card-modal-title">{title}</h3>}
          </div>
          <button type="button" className="card-modal-close" onClick={onClose} aria-label="Close">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" aria-hidden="true">
              <path d="M6 6l12 12M18 6L6 18" />
            </svg>
          </button>
        </div>
        <div className="card-modal-body">{children}</div>
      </div>
    </div>
  );
}

function InfoCard({
  title,
  summary,
  detail,
  children,
  className = "",
  variant = "default",
  interactive = false,
  modalWide = false,
}) {
  const [open, setOpen] = useState(false);
  const [modalKey, setModalKey] = useState(0);
  const hasDetail = Boolean(detail);

  function showModal() {
    if (!hasDetail) return;
    setModalKey((value) => value + 1);
    setOpen(true);
  }

  function openModal(event) {
    if (!hasDetail) return;
    if (event?.target?.closest?.(".info-card-skip, button, a, input, select, textarea, label, .map, .chart, .table-wrap, table")) {
      return;
    }
    showModal();
  }

  const clickable = hasDetail && !interactive;
  const modalPreview =
    children ||
    (variant === "stat" ? (
      <>
        <span className="muted">{title}</span>
        {typeof summary === "string" || typeof summary === "number" ? (
          <strong>{summary}</strong>
        ) : (
          <div className="stat-custom-value">{summary}</div>
        )}
      </>
    ) : null);

  return (
    <>
      <div
        className={`card info-card${clickable ? " info-card--clickable" : ""} ${className}`}
        onClick={clickable ? openModal : undefined}
        onKeyDown={
          clickable
            ? (event) => {
                if (event.key === "Enter" || event.key === " ") {
                  event.preventDefault();
                  openModal(event);
                }
              }
            : undefined
        }
        role={clickable ? "button" : undefined}
        tabIndex={clickable ? 0 : undefined}
      >
        {variant === "stat" ? (
          <>
            <span className="muted">{title}</span>
            {typeof summary === "string" || typeof summary === "number" ? (
              <strong>{summary}</strong>
            ) : (
              <div className="stat-custom-value">{summary}</div>
            )}
          </>
        ) : (
          <>
            {title && <h3>{title}</h3>}
            {summary && <p className="muted card-summary">{summary}</p>}
          </>
        )}
        {children}
        {hasDetail && interactive && (
          <button
            type="button"
            className="card-info-btn info-card-skip"
            onClick={(event) => {
              event.stopPropagation();
              showModal();
            }}
          >
            Expand
          </button>
        )}
        {hasDetail && !interactive && <span className="card-more-hint">Click to expand</span>}
      </div>
      {open && (
        <CardModal title={title} wide={modalWide || interactive} onClose={() => setOpen(false)}>
          {modalPreview && (
            <section className="card-modal-section card-modal-section--preview" aria-label="Preview">
              <div className="card-modal-section-head">
                <span className="card-modal-section-badge">Preview</span>
              </div>
              <div
                key={modalKey}
                className={`card-modal-preview info-card-skip${variant === "stat" ? " card-modal-preview--stat" : ""}`}
              >
                {modalPreview}
              </div>
            </section>
          )}
          <section className="card-modal-section card-modal-section--about" aria-label="About">
            <div className="card-modal-section-head">
              <span className="card-modal-section-badge card-modal-section-badge--about">About this view</span>
            </div>
            <div className="card-modal-description">
              {typeof detail === "string" ? <p>{detail}</p> : detail}
            </div>
          </section>
        </CardModal>
      )}
    </>
  );
}

function Stat({ label, value, detail }) {
  return (
    <InfoCard
      variant="stat"
      className="stat"
      title={label}
      summary={value}
      detail={detail}
    />
  );
}

function ConfusionMatrixView({ labels, matrix }) {
  if (!matrix?.length || !labels?.length) {
    return <p className="muted">Confusion matrix will appear after a successful model training run.</p>;
  }

  const maxVal = Math.max(...matrix.flat(), 1);

  return (
    <div className="table-wrap confusion-matrix-wrap">
      <table className="confusion-matrix">
        <thead>
          <tr>
            <th>Actual \ Predicted</th>
            {labels.map((label) => (
              <th key={label} title={label}>
                {label.length > 14 ? `${label.slice(0, 13)}…` : label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, rowIndex) => (
            <tr key={labels[rowIndex]}>
              <th title={labels[rowIndex]}>
                {labels[rowIndex].length > 14 ? `${labels[rowIndex].slice(0, 13)}…` : labels[rowIndex]}
              </th>
              {row.map((value, colIndex) => (
                <td
                  key={`${rowIndex}-${colIndex}`}
                  style={{
                    backgroundColor: `color-mix(in srgb, var(--blue) ${Math.round(12 + (value / maxVal) * 68)}%, transparent)`,
                  }}
                  title={`Actual: ${labels[rowIndex]}, Predicted: ${labels[colIndex]}, Count: ${value}`}
                >
                  {value}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function DatasetUploadCard({ onUpload, uploading, fileName, statusMessage, hasData }) {
  const inputRef = useRef(null);
  const [dragOver, setDragOver] = useState(false);
  const suppressCardClickRef = useRef(false);

  function handleFiles(fileList) {
    const file = fileList?.[0];
    if (!file || uploading) return;
    onUpload(file);
    if (inputRef.current) inputRef.current.value = "";
  }

  function openPicker() {
    if (uploading) return;
    inputRef.current?.click();
  }

  function onDragOver(event) {
    event.preventDefault();
    if (!uploading) setDragOver(true);
  }

  function onDragLeave(event) {
    event.preventDefault();
    setDragOver(false);
  }

  function onDrop(event) {
    event.preventDefault();
    setDragOver(false);
    handleFiles(event.dataTransfer.files);
  }

  return (
    <div
      className={`dataset-upload-card${dragOver ? " drag-over" : ""}${uploading ? " uploading" : ""}${hasData ? " has-data" : ""}`}
      role="button"
      tabIndex={0}
      onClick={(event) => {
        if (suppressCardClickRef.current) {
          suppressCardClickRef.current = false;
          return;
        }
        const target = event.target;
        if (target instanceof Element && target.closest("button, input, a, label")) return;
        openPicker();
      }}
      onKeyDown={(event) => {
        if (uploading) return;
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          openPicker();
        }
      }}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      onDrop={onDrop}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".csv,.xlsx"
        hidden
        onChange={(event) => handleFiles(event.target.files)}
      />

      <div className="upload-card-icon" aria-hidden="true">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
          <path d="M12 16V4" />
          <path d="M8 8l4-4 4 4" />
          <path d="M4 17v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2" />
        </svg>
      </div>

      <span className="upload-card-eyebrow">Dataset</span>
      <h3>Upload Crime Dataset</h3>
      <p className="upload-card-copy">
        Drag and drop your raw Chicago crime file here, or choose a file from your computer.
      </p>

      <div className="upload-format-badges">
        <span className="pill">CSV</span>
        <span className="pill">Excel (.xlsx)</span>
      </div>

      <button
        type="button"
        className="primary upload-card-button"
        onClick={(event) => {
          event.preventDefault();
          event.stopPropagation();
          suppressCardClickRef.current = true;
          openPicker();
        }}
        disabled={uploading}
      >
        {uploading ? "Cleaning dataset..." : hasData ? "Replace dataset" : "Choose file"}
      </button>

      <p className="upload-card-note muted">Missing values, invalid dates, and duplicate rows are cleaned automatically.</p>

      {uploading && (
        <div className="upload-progress" role="status">
          <span className="upload-spinner" />
          Processing and validating records...
        </div>
      )}

      {!uploading && fileName && (
        <p className="upload-file-name">
          <strong>{hasData ? "Loaded file" : "Selected file"}:</strong> {fileName}
        </p>
      )}

      {!uploading && statusMessage && (
        <p className={`upload-status${statusMessage.includes("failed") ? " upload-status-error" : ""}`}>
          {statusMessage}
        </p>
      )}
    </div>
  );
}

function ChartView({ type, labels, datasets, indexAxis = "x", xLabel = "", yLabel = "", valueDecimals = 0 }) {
  const [tooltip, setTooltip] = useState(null);
  const isDark = useSystemTheme();
  const { grid: gridColor, pieStroke } = chartTheme();
  const width = 900;
  const height = 310;
  const values = datasets.flatMap((dataset) => dataset.data);
  const maxValue = Math.max(...values, 1);
  const scaleMax = maxValue * 1.12;
  const showTooltip = (event, text) => {
    setTooltip({ x: event.clientX + 12, y: event.clientY + 12, text });
  };
  const hideTooltip = () => setTooltip(null);

  function withTooltip(label, value, extra = "") {
    const suffix = extra ? ` ${extra}` : "";
    return `${label}: ${formatChartValue(value, valueDecimals)}${suffix}`;
  }

  if (type === "doughnut" || type === "pie") {
    const dataset = datasets[0];
    const total = dataset.data.reduce((sum, value) => sum + value, 0) || 1;
    const pieCx = 180;
    const pieCy = 150;
    const pieOuter = 132;
    const pieInner = type === "doughnut" ? 64 : 0;
    const pieLabelRadius = pieInner === 0 ? pieOuter * 0.62 : (pieOuter + pieInner) / 2;

    const slices = labels.map((label, index) => {
      const value = Number(dataset.data[index] || 0);
      const start = dataset.data.slice(0, index).reduce((sum, item) => sum + Number(item || 0), 0);
      const end = start + value;
      const midAngle = ((start + value / 2) / total) * Math.PI * 2 - Math.PI / 2;
      const arcSpan = (value / total) * Math.PI * 2;
      return {
        label,
        value,
        index,
        start,
        end,
        pct: (value / total) * 100,
        midAngle,
        arcSpan,
      };
    });

    function arcPath(slice) {
      const start = (slice.start / total) * Math.PI * 2 - Math.PI / 2;
      const end = (slice.end / total) * Math.PI * 2 - Math.PI / 2;
      const largeArc = end - start > Math.PI ? 1 : 0;
      const sx = pieCx + pieOuter * Math.cos(start);
      const sy = pieCy + pieOuter * Math.sin(start);
      const ex = pieCx + pieOuter * Math.cos(end);
      const ey = pieCy + pieOuter * Math.sin(end);
      const isx = pieCx + pieInner * Math.cos(end);
      const isy = pieCy + pieInner * Math.sin(end);
      const iex = pieCx + pieInner * Math.cos(start);
      const iey = pieCy + pieInner * Math.sin(start);

      if (pieInner === 0) {
        return `M ${pieCx} ${pieCy} L ${sx} ${sy} A ${pieOuter} ${pieOuter} 0 ${largeArc} 1 ${ex} ${ey} Z`;
      }

      return `M ${sx} ${sy} A ${pieOuter} ${pieOuter} 0 ${largeArc} 1 ${ex} ${ey} L ${isx} ${isy} A ${pieInner} ${pieInner} 0 ${largeArc} 0 ${iex} ${iey} Z`;
    }

    function sliceShowsLabel(slice) {
      return slice.pct >= 7 && slice.arcSpan >= 0.38;
    }

    return (
      <div className="chart-shell">
        {tooltip && <div className="hover-tooltip" style={{ left: tooltip.x, top: tooltip.y }}>{tooltip.text}</div>}
        <div className="svg-chart pie-layout">
        <svg viewBox="0 0 360 300" preserveAspectRatio="xMidYMid meet" role="img" className="pie-donut-svg">
          {slices.map((slice) => (
            <path
              className="chart-hotspot"
              key={slice.label}
              d={arcPath(slice)}
              fill={dataset.backgroundColor[slice.index]}
              stroke={pieStroke}
              strokeWidth="2"
              onMouseMove={(event) => showTooltip(
                event,
                `${slice.label}: ${formatChartValue(slice.value, valueDecimals)} (${slice.pct.toFixed(1)}%)`,
              )}
              onMouseLeave={hideTooltip}
            />
          ))}
          {slices.filter(sliceShowsLabel).map((slice) => {
            const x = pieCx + pieLabelRadius * Math.cos(slice.midAngle);
            const y = pieCy + pieLabelRadius * Math.sin(slice.midAngle);
            const valueLabel = formatChartValue(slice.value, valueDecimals);
            return (
              <g
                key={`${slice.label}-value`}
                className="pie-slice-label-group"
                transform={`translate(${x}, ${y})`}
              >
                <text textAnchor="middle" dominantBaseline="central" className="pie-slice-label">
                  {valueLabel}
                </text>
              </g>
            );
          })}
        </svg>
        <div className="legend-list">
          {slices.map((slice) => (
            <span
              key={slice.label}
              className="legend-item-with-value"
              onMouseMove={(event) => showTooltip(
                event,
                `${slice.label}: ${formatChartValue(slice.value, valueDecimals)} (${slice.pct.toFixed(1)}%)`,
              )}
              onMouseLeave={hideTooltip}
            >
              <i style={{ background: dataset.backgroundColor[slice.index] }} />
              <span className="legend-item-text">
                <strong>{slice.label}</strong>
                <em>{formatChartValue(slice.value, valueDecimals)} · {slice.pct.toFixed(1)}%</em>
              </span>
            </span>
          ))}
        </div>
      </div>
      </div>
    );
  }

  if (type === "line") {
    const palette = ["#2563eb", "#0f766e", "#dc2626", "#9333ea"];
    const allValues = datasets.flatMap((dataset) => dataset.data);
    const lineScaleMax = Math.max(...allValues, 1) * 1.12;
    const showAllPointLabels = labels.length <= 18;

    return (
      <div className="chart-shell">
        {tooltip && <div className="hover-tooltip" style={{ left: tooltip.x, top: tooltip.y }}>{tooltip.text}</div>}
        {datasets.length > 1 && (
          <div className="chart-legend">
            {datasets.map((dataset, index) => (
              <span key={dataset.label}>
                <i style={{ background: dataset.borderColor || palette[index % palette.length] }} />
                {dataset.label}
              </span>
            ))}
          </div>
        )}
        <svg className="svg-chart" viewBox={`0 0 ${width} ${height}`} role="img">
        <line x1="45" y1="20" x2="45" y2={height - 42} stroke={gridColor} />
        <line x1="45" y1={height - 42} x2={width - 20} y2={height - 42} stroke={gridColor} />
        {yLabel && <text className="axis-label" x="15" y={height / 2} transform={`rotate(-90 15 ${height / 2})`} textAnchor="middle">{yLabel}</text>}
        {xLabel && <text className="axis-label" x={width / 2} y={height - 2} textAnchor="middle">{xLabel}</text>}
        {datasets.map((dataset, datasetIndex) => {
          const stroke = dataset.borderColor || palette[datasetIndex % palette.length];
          const points = dataset.data.map((value, index) => {
            const x = 50 + (index / Math.max(dataset.data.length - 1, 1)) * (width - 90);
            const y = height - 45 - (value / lineScaleMax) * (height - 80);
            return `${x},${y}`;
          });
          return (
            <g key={dataset.label}>
              <polyline points={points.join(" ")} fill="none" stroke={stroke} strokeWidth="3" pointerEvents="none" />
              {dataset.data.map((value, index) => {
                const x = 50 + (index / Math.max(dataset.data.length - 1, 1)) * (width - 90);
                const y = height - 45 - (value / lineScaleMax) * (height - 80);
                const showLabel = showAllPointLabels || index % Math.max(Math.ceil(labels.length / 8), 1) === 0;
                return (
                  <g key={`${dataset.label}-${labels[index]}`}>
                    <circle
                      className="chart-hotspot"
                      cx={x}
                      cy={y}
                      r="6"
                      fill={stroke}
                      onMouseMove={(event) => showTooltip(event, withTooltip(`${dataset.label} ${labels[index]}`, value, "cases"))}
                      onMouseLeave={hideTooltip}
                    />
                    {showLabel && (
                      <text
                        x={x}
                        y={y - 12}
                        textAnchor="middle"
                        className="chart-value-label"
                      >
                        {formatChartValue(value, valueDecimals || 1)}
                      </text>
                    )}
                  </g>
                );
              })}
            </g>
          );
        })}
        {labels.map((label, index) => (
          index % Math.max(Math.ceil(labels.length / 8), 1) === 0 ? <text key={label} x={50 + (index / Math.max(labels.length - 1, 1)) * (width - 90)} y={height - 18} textAnchor="middle">{label}</text> : null
        ))}
      </svg>
      </div>
    );
  }

  const vertical = indexAxis === "y";
  const datasetCount = datasets.length;
  return (
    <div className="chart-shell">
      {tooltip && <div className="hover-tooltip" style={{ left: tooltip.x, top: tooltip.y }}>{tooltip.text}</div>}
      {datasets.length > 1 && (
        <div className="chart-legend">
          {datasets.map((dataset) => (
            <span key={dataset.label}>
              <i style={{ background: dataset.backgroundColor || "#2563eb" }} />
              {dataset.label}
            </span>
          ))}
        </div>
      )}
      <svg className="svg-chart" viewBox={`0 0 ${width} ${height}`} role="img">
      <line x1="45" y1="20" x2="45" y2={height - 52} stroke={gridColor} />
      <line x1="45" y1={height - 52} x2={width - 20} y2={height - 52} stroke={gridColor} />
      {yLabel && <text className="axis-label" x="15" y={height / 2} transform={`rotate(-90 15 ${height / 2})`} textAnchor="middle">{yLabel}</text>}
      {xLabel && <text className="axis-label" x={width / 2} y={height - 2} textAnchor="middle">{xLabel}</text>}
      {!vertical &&
        labels.map((label, labelIndex) => {
          const groupWidth = (width - 90) / labels.length;
          return datasets.map((dataset, datasetIndex) => {
            const barWidth = Math.max(groupWidth / datasetCount - 8, 8);
            const value = dataset.data[labelIndex] || 0;
            const barHeight = (value / scaleMax) * (height - 88);
            const x = 52 + labelIndex * groupWidth + datasetIndex * (barWidth + 4);
            const y = height - 52 - barHeight;
            return (
              <g key={`${dataset.label}-${label}`}>
                <rect
                  className="chart-hotspot"
                  x={x}
                  y={y}
                  width={barWidth}
                  height={barHeight}
                  rx="4"
                  fill={dataset.backgroundColor || "#2563eb"}
                  onMouseMove={(event) => showTooltip(event, withTooltip(`${dataset.label} ${label}`, value))}
                  onMouseLeave={hideTooltip}
                />
                {barHeight > 0 && (
                  <text
                    x={x + barWidth / 2}
                    y={Math.max(y - 6, 16)}
                    textAnchor="middle"
                    className="chart-value-label"
                  >
                    {formatChartValue(value, valueDecimals)}
                  </text>
                )}
              </g>
            );
          });
        })}
      {vertical &&
        labels.map((label, index) => {
          const rowHeight = (height - 72) / labels.length;
          const value = datasets[0].data[index] || 0;
          const barWidth = (value / scaleMax) * (width - 250);
          const barY = 22 + index * rowHeight;
          return (
            <g key={label}>
              <text x="48" y={34 + index * rowHeight} dominantBaseline="middle">{label}</text>
              <rect
                className="chart-hotspot"
                x="170"
                y={barY}
                width={barWidth}
                height={Math.max(rowHeight - 8, 8)}
                rx="4"
                fill={datasets[0].backgroundColor || "#2563eb"}
                onMouseMove={(event) => showTooltip(event, withTooltip(label, value, "cases"))}
                onMouseLeave={hideTooltip}
              />
              <text
                x={170 + barWidth + 8}
                y={barY + Math.max(rowHeight - 8, 8) / 2}
                dominantBaseline="middle"
                className="chart-value-label"
              >
                {formatChartValue(value, valueDecimals)}
              </text>
            </g>
          );
        })}
      {!vertical &&
        labels.map((label, index) => (
          index % Math.ceil(labels.length / 8) === 0 ? <text key={label} x={60 + index * ((width - 90) / labels.length)} y={height - 20} textAnchor="middle">{label.length > 13 ? `${label.slice(0, 12)}...` : label}</text> : null
        ))}
    </svg>
    </div>
  );
}

function Section({ eyebrow, title, description, children }) {
  return (
    <section className="section-block">
      <div className="section-heading">
        {eyebrow && <span>{eyebrow}</span>}
        <div>
          <h3>{title}</h3>
          {description && <p>{description}</p>}
        </div>
      </div>
      {children}
    </section>
  );
}

function PieChartView({ rows }) {
  const palette = ["#2563eb", "#0f766e", "#dc2626", "#9333ea", "#ea580c", "#0891b2", "#4f46e5", "#65a30d"];
  const topRows = rows.slice(0, 7);
  const otherTotal = rows.slice(7).reduce((sum, row) => sum + row.count, 0);
  const pieRows = otherTotal > 0 ? [...topRows, { Description: "Other", count: otherTotal }] : topRows;

  return (
    <ChartView
      type="doughnut"
      labels={pieRows.map((row) => row.Description)}
      datasets={[
        {
          label: "Cases",
          data: pieRows.map((row) => row.count),
          backgroundColor: pieRows.map((_, index) => palette[index % palette.length]),
          borderColor: "#ffffff",
          borderWidth: 2,
        },
      ]}
    />
  );
}

function riskColor(level) {
  if (level === "HIGH") return "#dc2626";
  if (level === "MEDIUM") return "#eab308";
  return "#16a34a";
}

function heatIntensity(point) {
  if (point.Risk_Level === "HIGH") return 1;
  if (point.Risk_Level === "MEDIUM") return 0.72;
  if (point.Risk_Level === "LOW") return 0.45;
  if (point.Is_Correct === false) return 1;
  if (point.Is_Correct === true) return 0.42;
  if (point.Risk_Score !== undefined && point.Risk_Score !== null) {
    const score = Number(point.Risk_Score);
    if (Number.isFinite(score)) return Math.min(Math.max(score / 100, 0.3), 1);
  }
  return 0.62;
}

const HEAT_GRADIENT = {
  0.2: "#16a34a",
  0.45: "#84cc16",
  0.65: "#eab308",
  0.82: "#f97316",
  1.0: "#dc2626",
};

function MapView({ points, color = "#0f766e", heatmap = true }) {
  const nodeRef = useRef(null);
  const mapRef = useRef(null);
  const tileRef = useRef(null);
  const layerRef = useRef(null);

  useEffect(() => {
    if (!nodeRef.current || !window.L) return;

    if (!mapRef.current) {
      mapRef.current = L.map(nodeRef.current, { scrollWheelZoom: false }).setView([41.8781, -87.6298], 10);
    }

    if (tileRef.current) {
      tileRef.current.remove();
    }

    tileRef.current = L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      attribution: "&copy; OpenStreetMap contributors",
    }).addTo(mapRef.current);

    if (layerRef.current) {
      mapRef.current.removeLayer(layerRef.current);
      layerRef.current = null;
    }

    const valid = (points || []).filter((point) => point.Latitude && point.Longitude).slice(0, 2000);
    const useHeat = heatmap && typeof L.heatLayer === "function";

    if (useHeat && valid.length) {
      const heatData = valid.map((point) => [point.Latitude, point.Longitude, heatIntensity(point)]);
      layerRef.current = L.heatLayer(heatData, {
        radius: 26,
        blur: 20,
        maxZoom: 17,
        minOpacity: 0.38,
        gradient: HEAT_GRADIENT,
      }).addTo(mapRef.current);
    } else {
      layerRef.current = L.layerGroup().addTo(mapRef.current);
      valid.slice(0, 900).forEach((point) => {
        const markerColor = point.Risk_Level
          ? riskColor(point.Risk_Level)
          : typeof color === "function"
            ? color(point)
            : color;
        const tooltipParts = [
          point.District_Name ? `District: ${point.District_Name}${point.District ? ` (${point.District})` : ""}` : null,
          point.Prediction_Month ? `Prediction Month: ${point.Prediction_Month}` : null,
          point.Prediction_Year ? `Prediction Year: ${point.Prediction_Year}` : null,
          point.Expected_Crime_Count !== undefined ? `Expected Crime Count: ${point.Expected_Crime_Count}` : null,
          point.Risk_Score !== undefined ? `Risk Score: ${point.Risk_Score}` : null,
          point.Risk_Level ? `Risk Level: ${point.Risk_Level}` : null,
          point.Predicted_District ? `Predicted: ${point.Predicted_District}` : null,
          point.Is_Correct !== undefined ? `Status: ${point.Is_Correct ? "Correct" : "Wrong"}` : null,
          point.hour !== undefined ? `Hour: ${point.hour}:00` : null,
        ].filter(Boolean);

        L.circleMarker([point.Latitude, point.Longitude], {
          radius: point.Risk_Level ? 10 : 7,
          color: markerColor,
          fillColor: markerColor,
          fillOpacity: 0.75,
          weight: 1,
        })
          .bindTooltip(tooltipParts.join("<br>"), { sticky: true, direction: "top", opacity: 0.95 })
          .addTo(layerRef.current);
      });
    }

    if (valid.length) {
      const bounds = L.latLngBounds(valid.map((point) => [point.Latitude, point.Longitude]));
      mapRef.current.fitBounds(bounds, { padding: [18, 18] });
    }

    const resizeTimers = [80, 250].map((delay) => setTimeout(() => mapRef.current?.invalidateSize(), delay));
    return () => resizeTimers.forEach(clearTimeout);
  }, [points, color, heatmap]);

  return (
    <div className="map-wrap info-card-skip">
      {heatmap && (
        <p className="muted map-heatmap-note">
          Heatmap density — cooler greens indicate lower concentration; yellow and red indicate stronger hotspots.
        </p>
      )}
      <div className="map" ref={nodeRef} />
    </div>
  );
}

function DataTable({ rows }) {
  if (!rows || rows.length === 0) {
    return <p className="muted">No rows available.</p>;
  }

  const columns = Object.keys(rows[0]);

  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            {columns.map((column) => (
              <th key={column}>{column}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={index}>
              {columns.map((column) => (
                <td key={column}>{String(row[column] ?? "")}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function CleaningReport({ cleaning }) {
  if (!cleaning) return null;

  const issueRows = (cleaning.issues || []).filter((issue) => issue.Count > 0);

  return (
    <Section
      eyebrow="Clean"
      title="Raw Dataset Cleaning Report"
      description="The uploaded raw file is inspected for missing values, invalid dates, invalid coordinates, duplicate rows, placeholder times, and unusable district values before analysis."
    >
      <div className="grid cleaning-grid">
        <Stat label="Original rows" value={formatNumber(cleaning.originalRows)} detail={CARD_DETAILS.statOriginalRows} />
        <Stat label="Clean usable rows" value={formatNumber(cleaning.cleanRows)} detail={CARD_DETAILS.statCleanRows} />
        <Stat label="Rows removed" value={formatNumber(cleaning.removedRows)} detail={CARD_DETAILS.statRowsRemoved} />
        <Stat
          label="Issues detected"
          value={formatNumber(issueRows.reduce((sum, row) => sum + row.Count, 0))}
          detail={CARD_DETAILS.statIssues}
        />
      </div>
      <InfoCard
        className="compact-table-card"
        title="Detected Data Quality Issues"
        detail="Each row lists a data-quality problem found in the raw upload (missing values, invalid coordinates, duplicates, etc.) and how many records were affected before cleaning."
        interactive
      >
        {issueRows.length ? <DataTable rows={issueRows} /> : <p className="muted">No major data quality issues were detected.</p>}
      </InfoCard>
    </Section>
  );
}

function Home({ onUpload, uploading, message, hasData, fileName }) {
  return (
    <section className="hero-layout">
      <div className="hero-content">
        <h2>A Hybrid Machine Learning Framework for Geospatial Cyber Crime Prediction and Demographic Pattern Analysis</h2>
        <p>
          Upload the raw Chicago crime dataset. The system cleans missing and invalid records, then prepares the usable
          dataset for spatial analysis, temporal analysis, and machine learning prediction.
        </p>
        <div className="hero-meta">
          <span className="pill">Geospatial analysis</span>
          <span className="pill">Temporal patterns</span>
          <span className="pill">Machine learning</span>
          <span className="pill">Future hotspot prediction</span>
        </div>
      </div>

      <DatasetUploadCard
        onUpload={onUpload}
        uploading={uploading}
        fileName={fileName}
        statusMessage={message}
        hasData={hasData}
      />
    </section>
  );
}

function Overview({ data }) {
  if (!data) {
    return <div className="message">Upload a raw CSV or Excel dataset. The app will clean it before opening the dashboard.</div>;
  }

  return (
    <div className="grid">
      <div className="grid stats">
        <Stat label="Original rows" value={formatNumber(data.cleaning?.originalRows || data.rows)} detail={CARD_DETAILS.statOriginalRows} />
        <Stat label="Clean rows" value={formatNumber(data.rows)} detail={CARD_DETAILS.statCleanRows} />
        <Stat label="Crime categories" value={formatNumber(data.summary.crimeTypes)} detail={CARD_DETAILS.statCrimeTypes} />
        <Stat label="Rows removed" value={formatNumber(data.cleaning?.removedRows || 0)} detail={CARD_DETAILS.statRowsRemoved} />
      </div>

      <CleaningReport cleaning={data.cleaning} />

      <Section
        eyebrow="Overview"
        title="Crime Distribution"
        description="This section summarizes how cybercrime cases are distributed across major crime categories in the cleaned dataset."
      >
        <div className="grid overview-grid">
          <InfoCard title="Crime Category Share" detail={CARD_DETAILS.crimeCategoryShare} interactive>
            <div className="chart pie-chart">
              <PieChartView rows={data.charts.descriptionCounts} />
            </div>
          </InfoCard>

          <InfoCard title="Top Crime Categories" detail={CARD_DETAILS.topCrimeCategories} interactive>
            <div className="chart">
              <ChartView
                type="bar"
                labels={data.charts.descriptionCounts.map((row) => row.Description)}
                datasets={[{ label: "Cases", data: data.charts.descriptionCounts.map((row) => row.count), backgroundColor: "#2563eb" }]}
                xLabel="Crime category"
                yLabel="Number of cases"
              />
            </div>
          </InfoCard>
        </div>
      </Section>

      <Section
        eyebrow="Dataset"
        title="Cleaned Dataset Overview"
        description="A compact preview of the cleaned and usable records generated from the uploaded raw dataset."
      >
        <InfoCard title="Cleaned Dataset Preview" detail={CARD_DETAILS.cleanedPreview} interactive>
          <p className="muted table-note">
            {data.raw.columns.length} columns detected after cleaning. Date range: {data.summary.dateMin || "-"} to {data.summary.dateMax || "-"}.
            Showing the first 20 usable rows.
          </p>
          <DataTable rows={data.raw.preview} />
        </InfoCard>
      </Section>

      <Section
        eyebrow="Reference"
        title="Chicago Police Districts"
        description="District number and district name reference used to interpret spatial and prediction outputs."
      >
        <InfoCard className="compact-table-card" title="Chicago Police Districts" detail={CARD_DETAILS.districtReference} interactive>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>No</th>
                  <th>District</th>
                </tr>
              </thead>
              <tbody>
                {data.districtReference.map((row) => (
                  <tr key={row["District No"]}>
                    <td>{row["District No"]}</td>
                    <td>{row["District Name"]}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </InfoCard>
      </Section>
    </div>
  );
}

function Analysis({ data }) {
  const [crime, setCrime] = useState("");
  const [hour, setHour] = useState("");
  const [debouncedHour, setDebouncedHour] = useState("");
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    if (data?.crimeTypes?.length && !crime) {
      setCrime(data.crimeTypes[0]);
    }
  }, [data, crime]);

  useEffect(() => {
    const timer = setTimeout(() => setDebouncedHour(hour), 400);
    return () => clearTimeout(timer);
  }, [hour]);

  useEffect(() => {
    if (!crime) return;
    setLoading(true);
    setError("");
    const query = new URLSearchParams({ crime });
    if (debouncedHour !== "") query.set("hour", debouncedHour);
    fetch(`/api/analysis?${query.toString()}`)
      .then((res) => res.json().then((body) => (res.ok ? body : Promise.reject(body))))
      .then(setAnalysis)
      .catch((err) => setError(err.error || "Could not load analysis."))
      .finally(() => setLoading(false));
  }, [crime, debouncedHour]);

  if (!data) {
    return <div className="message">Upload a raw dataset first. The app will clean it before analysis.</div>;
  }

  return (
    <div className="grid">
      <div className="controls">
        <div className="field">
          <label>Crime category</label>
          <select value={crime} onChange={(event) => setCrime(event.target.value)}>
            {data.crimeTypes.map((item) => (
              <option key={item}>{item}</option>
            ))}
          </select>
        </div>
        <div className="field">
          <label>Hour</label>
          <input min="0" max="23" type="number" value={hour} placeholder="Peak" onChange={(event) => setHour(event.target.value)} />
        </div>
      </div>

      {error && <div className="message error">{error}</div>}
      {loading && <div className="message">Loading analysis...</div>}

      {analysis && (
        <>
          <div className="grid stats">
            <Stat label="Selected records" value={formatNumber(analysis.records)} detail={CARD_DETAILS.analysisRecords} />
            <Stat label="Peak hour" value={`${analysis.peakHour}:00`} detail={CARD_DETAILS.analysisPeakHour} />
            <Stat label="Viewed hour" value={`${analysis.selectedHour}:00`} detail={CARD_DETAILS.analysisViewedHour} />
            <Stat label="Mapped points" value={formatNumber(analysis.mapPoints.length)} detail={CARD_DETAILS.analysisMapPoints} />
          </div>

          {analysis.highestRiskDistrict && (
            <InfoCard className="highest-risk-card" detail={CARD_DETAILS.highestRiskDistrict}>
              <span className="muted">Highest Risk District</span>
              <strong>
                {analysis.highestRiskDistrict.highest_risk_district
                  ? `District ${analysis.highestRiskDistrict.highest_risk_district}${analysis.highestRiskDistrict.district_name ? ` - ${analysis.highestRiskDistrict.district_name}` : ""}`
                  : analysis.highestRiskDistrict.district_name || "—"}
              </strong>
              <p className="highest-risk-meta">
                {formatNumber(analysis.highestRiskDistrict.crime_count)} crimes | {Number(analysis.highestRiskDistrict.risk_percentage).toFixed(1)}% risk
              </p>
            </InfoCard>
          )}

          <Section
            eyebrow="1"
            title="District-wise Analysis"
            description="Shows which police districts have the highest number of incidents for the selected crime category."
          >
            <InfoCard title="District-wise Incident Counts" detail={CARD_DETAILS.districtWiseChart} interactive>
              <div className="chart chart-medium">
                <ChartView
                  type="bar"
                  indexAxis="y"
                  labels={analysis.topDistricts.map((row) => row.district)}
                  datasets={[{ label: "Cases", data: analysis.topDistricts.map((row) => row.count), backgroundColor: "#2563eb" }]}
                  xLabel="Number of cases"
                  yLabel="District"
                />
              </div>
            </InfoCard>
          </Section>

          <Section
            eyebrow="2"
            title="Geospatial Analysis"
            description="Plots incident locations as a heatmap to reveal geographical clustering and hotspot areas for the selected crime."
          >
            <InfoCard title="Geospatial Crime Heatmap" detail={CARD_DETAILS.spatialHeatmap} interactive modalWide>
              <MapView points={analysis.mapPoints} heatmap />
            </InfoCard>
          </Section>

          <Section
            eyebrow="3"
            title="Temporal Analysis"
            description="Compares incident frequency across all 24 hours to identify peak reporting periods."
          >
            <InfoCard title="Hourly Incident Distribution" detail={CARD_DETAILS.temporalChart} interactive>
              <div className="chart chart-medium">
                <ChartView
                  type="bar"
                  labels={analysis.hourlyCounts.map((row) => `${row.hour}:00`)}
                  datasets={[{ label: "Cases", data: analysis.hourlyCounts.map((row) => row.count), backgroundColor: "#0f766e" }]}
                  xLabel="Hour of day"
                  yLabel="Number of cases"
                />
              </div>
            </InfoCard>
          </Section>

          <Section
            eyebrow="4"
            title="Time-wise Geospatial Analysis"
            description="Filters the geospatial heatmap by the selected hour to compare how hotspots change over time."
          >
            <div className="grid two">
              <InfoCard title="Hour-filtered Heatmap" detail={CARD_DETAILS.hourSpatialHeatmap} interactive modalWide>
                <MapView points={analysis.hourMapPoints} color="#dc2626" heatmap />
              </InfoCard>
              <InfoCard
                className="compact-table-card"
                title="Selected-hour District Concentration"
                detail={CARD_DETAILS.hourDistrictTable}
                interactive
              >
                <DataTable rows={analysis.hourTopDistricts} />
              </InfoCard>
            </div>
          </Section>
        </>
      )}
    </div>
  );
}

const INITIAL_ML_SESSION = {
  crime: "",
  model: "Logistic Regression",
  result: null,
  error: "",
};

const INITIAL_FUTURE_SESSION = {
  crime: "",
  district: "",
  month: "8",
  year: "2026",
  activeModel: "",
  result: null,
  error: "",
  districts: [],
  crimeTypes: [],
  optionsLoaded: false,
};

function MachineLearning({ data, session, setSession }) {
  const { crime, model, result, error } = session;
  const [modelNames, setModelNames] = useState(DEFAULT_MODELS);
  const [loading, setLoading] = useState(false);

  const patchSession = (updates) => setSession((prev) => ({ ...prev, ...updates }));

  useEffect(() => {
    fetch("/api/models")
      .then((res) => res.json())
      .then((body) => {
        const availableModels = withoutRemovedModels(body.models);
        if (availableModels.length) setModelNames(availableModels);
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    if (data?.crimeTypes?.length && !crime) {
      setSession((prev) => ({ ...prev, crime: data.crimeTypes[0] }));
    }
  }, [data, crime, setSession]);

  useEffect(() => {
    if (modelNames.length && !modelNames.includes(model)) {
      setSession((prev) => ({ ...prev, model: modelNames[0] }));
    }
  }, [model, modelNames, setSession]);

  function train() {
    setLoading(true);
    patchSession({ error: "", result: null });
    const requestedModel = isRemovedModel(model) ? modelNames[0] : model;
    fetch("/api/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ crime, model: requestedModel }),
    })
      .then((res) => res.json().then((body) => (res.ok ? body : Promise.reject(body))))
      .then((body) => patchSession({ result: body, error: "" }))
      .catch((err) => patchSession({ error: err.error || "Training failed.", result: null }))
      .finally(() => setLoading(false));
  }

  if (!data) {
    return <div className="message">Upload a raw dataset first. The app will clean it before model training.</div>;
  }

  const comparisonRows = withoutRemovedModels(result?.comparison);
  const selectedMlModel = isRemovedModel(result?.selected?.model)
    ? comparisonRows[0]?.Model || modelNames[0] || ""
    : result?.selected?.model;

  return (
    <div className="grid">
      <div className="controls">
        <div className="field">
          <label>Crime category</label>
          <select value={crime} onChange={(event) => patchSession({ crime: event.target.value })}>
            {data.crimeTypes.map((item) => (
              <option key={item}>{item}</option>
            ))}
          </select>
        </div>
        <div className="field">
          <label>Model</label>
          <select value={model} onChange={(event) => patchSession({ model: event.target.value })}>
            {modelNames.map((item) => (
              <option key={item}>{item}</option>
            ))}
          </select>
        </div>
        <button className="primary" onClick={train} disabled={loading}>
          {loading ? "Training..." : "Train Model"}
        </button>
      </div>

      {error && <div className="message error">{error}</div>}

      {result && (
        <>
          <div className="grid stats">
            <Stat label="Selected model" value={selectedMlModel} detail={CARD_DETAILS.mlSelectedModel} />
            <Stat label="Accuracy" value={`${result.selected.accuracy.toFixed(2)}%`} detail={CARD_DETAILS.mlAccuracy} />
            <Stat label="Precision" value={`${(result.selected.precision ?? 0).toFixed(2)}%`} detail={CARD_DETAILS.mlPrecision} />
            <Stat label="Recall" value={`${(result.selected.recall ?? 0).toFixed(2)}%`} detail={CARD_DETAILS.mlRecall} />
            <Stat label="F1 score" value={`${(result.selected.f1 ?? 0).toFixed(2)}%`} detail={CARD_DETAILS.mlF1} />
            <Stat label="Crime category" value={result.crime} detail={CARD_DETAILS.mlCrimeCategory} />
          </div>

          <Section
            eyebrow="Model"
            title="Machine Learning Model Comparison"
            description="Weighted precision, recall, F1-score, and accuracy across all classifiers for the selected crime category."
          >
          <InfoCard title="Classifier Metrics Comparison" detail={CARD_DETAILS.mlModelComparison} interactive>
            <div className="chart">
              <ChartView
                type="bar"
                labels={comparisonRows.map((row) => shortModelName(row.Model))}
                valueDecimals={2}
                datasets={[
                  { label: "Accuracy", data: comparisonRows.map((row) => row.Accuracy), backgroundColor: "#2563eb" },
                  { label: "Precision", data: comparisonRows.map((row) => row.Precision), backgroundColor: "#9333ea" },
                  { label: "Recall", data: comparisonRows.map((row) => row.Recall), backgroundColor: "#dc2626" },
                  { label: "F1", data: comparisonRows.map((row) => row.F1), backgroundColor: "#0f766e" },
                ]}
                xLabel="Machine learning model"
                yLabel="Score (%)"
              />
            </div>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Model</th>
                    <th>Accuracy</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1</th>
                  </tr>
                </thead>
                <tbody>
                  {comparisonRows.map((row) => (
                    <tr key={row.Model}>
                      <td>{row.Model}</td>
                      <td>{row.Accuracy.toFixed(2)}%</td>
                      <td>{row.Precision.toFixed(2)}%</td>
                      <td>{row.Recall.toFixed(2)}%</td>
                      <td>{row.F1.toFixed(2)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </InfoCard>
          </Section>

          <Section
            eyebrow="Matrix"
            title="Confusion Matrix"
            description="Test-set confusion matrix for the selected model: actual districts (rows) versus predicted districts (columns)."
          >
            <InfoCard title="District Confusion Matrix" detail={CARD_DETAILS.mlConfusionMatrix} interactive>
              <ConfusionMatrixView
                labels={(result.confusionMatrix || result.selected.confusionMatrix)?.labels || []}
                matrix={(result.confusionMatrix || result.selected.confusionMatrix)?.matrix || []}
              />
            </InfoCard>
          </Section>

          <Section
            eyebrow="District"
            title="District Evaluation Report"
            description="Per-district precision, recall, F1-score, and support for the selected model (district-level breakdown)."
          >
          <InfoCard title="Per-district Classification Metrics" detail={CARD_DETAILS.mlDistrictReport} interactive>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>District</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1</th>
                    <th>Support</th>
                  </tr>
                </thead>
                <tbody>
                  {result.selected.report.map((row) => (
                    <tr key={row.District}>
                      <td>{row.District}</td>
                      <td>{row.Precision.toFixed(2)}%</td>
                      <td>{row.Recall.toFixed(2)}%</td>
                      <td>{row.F1.toFixed(2)}%</td>
                      <td>{row.Support}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </InfoCard>
          </Section>

          <Section
            eyebrow="Map"
            title="Predicted District Map"
            description="Incident locations for the selected model, colored by whether the predicted district matches the actual district."
          >
          <InfoCard title="Predicted District Map" detail={CARD_DETAILS.mlPredictionHeatmap} interactive modalWide>
            <p className="muted table-note">Green spots = correct predictions. Red spots = incorrect predictions.</p>
            <MapView points={result.predictionMap} color={(point) => (point.Is_Correct ? "#16a34a" : "#dc2626")} heatmap={false} />
          </InfoCard>
          </Section>
        </>
      )}
    </div>
  );
}

function RiskBadge({ level }) {
  const tone = (level || "LOW").toLowerCase();
  return <span className={`risk-badge risk-${tone}`}>{level || "LOW"}</span>;
}

function hotspotModelMetrics(result, modelName) {
  if (isRemovedModel(modelName)) return null;
  if (!result || !modelName) return null;
  const fromEvaluation = result.evaluation?.[modelName];
  if (fromEvaluation) return fromEvaluation;
  return withoutRemovedModels(result.comparison).find((row) => row.Model === modelName) || null;
}

function formatPredictedCrimes(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return "—";
  return numeric.toFixed(1);
}

async function readApiResponse(res) {
  let body = null;
  try {
    body = await res.json();
  } catch {
    throw new Error(`Server error (${res.status}). Restart python react_api.py, then hard-refresh the page (Ctrl+Shift+R).`);
  }
  if (!res.ok) {
    throw new Error(body?.error || `Request failed (${res.status})`);
  }
  return body;
}

function FutureCrimePrediction({ data, session, setSession }) {
  const monthOptions = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
  ];
  const {
    districts,
    crimeTypes,
    district,
    crime,
    month,
    year,
    activeModel,
    result,
    error,
    optionsLoaded,
  } = session;
  const [loading, setLoading] = useState(false);
  const [modelSwitching, setModelSwitching] = useState(false);
  const forecastRequestRef = useRef(0);

  const patchSession = (updates) => setSession((prev) => ({ ...prev, ...updates }));

  useEffect(() => {
    if (!data || optionsLoaded) return;
    fetch("/api/hotspot/options")
      .then((res) => res.json().then((body) => (res.ok ? body : Promise.reject(body))))
      .then((body) => {
        setSession((prev) => ({
          ...prev,
          optionsLoaded: true,
          districts: body.districts || [],
          crimeTypes: body.crimeTypes || data.crimeTypes || [],
          district: prev.district || String(body.defaultDistrict || 11),
          crime: prev.crime || body.defaultCrime || data.crimeTypes?.[0] || "",
          month: prev.month || String(body.defaultMonth || 8),
          year: prev.year || String(body.defaultYear || 2026),
          activeModel: prev.activeModel || body.selectedModel || "",
        }));
      })
      .catch((err) => patchSession({ error: err.error || "Could not load hotspot options." }));
  }, [data, optionsLoaded, setSession]);

  function handleCrimeChange(nextCrime) {
    setSession((prev) => ({
      ...prev,
      crime: nextCrime,
      result: prev.result?.crime === nextCrime ? prev.result : null,
      activeModel: prev.result?.crime === nextCrime ? prev.activeModel : "",
      error: "",
    }));
  }

  function runForecast({ modelOverride, switchOnly = false } = {}) {
    if (!district || !crime) {
      patchSession({ error: "Select a crime category and district first." });
      return;
    }

    const requestId = forecastRequestRef.current + 1;
    forecastRequestRef.current = requestId;
    setLoading(true);
    setModelSwitching(Boolean(switchOnly));
    patchSession({ error: "" });

    const resultFromSameCrime = result?.crime === crime ? result : null;
    const chosenModel = modelOverride || (switchOnly ? (resultFromSameCrime?.activeModel || resultFromSameCrime?.model) : undefined);

    fetch("/api/hotspot/forecast", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        district: Number(district),
        month: Number(month),
        year: Number(year),
        crime,
        model: chosenModel,
        lightweight: Boolean(switchOnly),
      }),
    })
      .then(readApiResponse)
      .then((body) => {
        if (forecastRequestRef.current !== requestId) return;
        patchSession({
          result: body,
          activeModel: body.activeModel || body.model || "",
          error: "",
        });
      })
      .catch((err) => {
        if (forecastRequestRef.current !== requestId) return;
        patchSession({ error: err.message || err.error || "Hotspot prediction failed." });
      })
      .finally(() => {
        if (forecastRequestRef.current !== requestId) return;
        setLoading(false);
        setModelSwitching(false);
      });
  }

  function predict() {
    runForecast();
  }

  function selectModel(modelName) {
    if (!result || !modelName || loading) return;
    if (isRemovedModel(modelName)) return;
    const currentModel = result.activeModel || result.model;
    if (modelName === currentModel) return;
    if (result.evaluation?.[modelName]?.Error) {
      patchSession({ error: `Model unavailable: ${result.evaluation[modelName].Error}` });
      return;
    }

    const requestId = forecastRequestRef.current + 1;
    forecastRequestRef.current = requestId;
    setModelSwitching(true);
    setLoading(true);
    patchSession({ error: "" });

    setSession((prev) => ({
      ...prev,
      activeModel: modelName,
      result: prev.result
        ? {
            ...prev.result,
            activeModel: modelName,
            model: modelName,
            selectedModelMetrics: hotspotModelMetrics(prev.result, modelName) || prev.result.selectedModelMetrics,
          }
        : prev.result,
    }));

    fetch("/api/hotspot/forecast", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        district: Number(district),
        month: Number(month),
        year: Number(year),
        crime,
        model: modelName,
        lightweight: true,
      }),
    })
      .then(readApiResponse)
      .then((body) => {
        if (forecastRequestRef.current !== requestId) return;
        setSession((prev) => ({
          ...prev,
          result: prev.result ? { ...prev.result, ...body } : body,
          activeModel: body.activeModel || body.model || modelName,
          error: "",
        }));
      })
      .catch((err) => {
        if (forecastRequestRef.current !== requestId) return;
        patchSession({ error: err.message || err.error || "Could not switch regression model." });
        setSession((prev) => ({
          ...prev,
          result: prev.result ? { ...prev.result, activeModel: currentModel, model: currentModel } : prev.result,
          activeModel: currentModel || "",
        }));
      })
      .finally(() => {
        if (forecastRequestRef.current !== requestId) return;
        setLoading(false);
        setModelSwitching(false);
      });
  }

  if (!data) {
    return <div className="message">Upload a raw dataset first. The app will clean it before future hotspot prediction.</div>;
  }

  const rawEvaluationRows = withoutRemovedModels(
    result?.comparison?.length ? result.comparison : result
      ? Object.entries(result.evaluation || {}).map(([model, metrics]) => ({ Model: model, ...metrics }))
      : []
  );
  const bestModel = isRemovedModel(result?.bestModel || result?.selectedModel)
    ? rawEvaluationRows[0]?.Model
    : result?.bestModel || result?.selectedModel;
  const displayModel = isRemovedModel(result?.activeModel || result?.model)
    ? rawEvaluationRows[0]?.Model || ""
    : result?.activeModel || result?.model || "";
  const bestModelR2 = Number(
    hotspotModelMetrics(result, bestModel)?.R2
    ?? result?.bestModelMetrics?.R2
    ?? 0,
  );
  const activeModelR2 = Number(
    hotspotModelMetrics(result, displayModel)?.R2
    ?? result?.selectedModelMetrics?.R2
    ?? 0,
  );
  const evaluationRows = rawEvaluationRows.map((row) => ({
    ...row,
    isBest: row.Model === bestModel,
    isActive: row.Model === displayModel,
    isUnavailable: Boolean(row.Error || result?.evaluation?.[row.Model]?.Error),
  }));

  return (
    <div className="grid">
      <Section
        eyebrow="Future"
        title="Future Crime Hotspot Prediction"
        description="Select a crime category, district, and month-year target. Train regression models on that crime type, then choose any evaluated model to update predictions and the actual vs predicted trend chart."
      >
        <div className="controls">
          <div className="field">
            <label>Crime category</label>
            <select value={crime} onChange={(event) => handleCrimeChange(event.target.value)}>
              {(crimeTypes.length ? crimeTypes : data.crimeTypes || []).map((item) => (
                <option key={item}>{item}</option>
              ))}
            </select>
          </div>
          <div className="field">
            <label>District</label>
            <select value={district} onChange={(event) => patchSession({ district: event.target.value })}>
              {districts.map((item) => (
                <option key={item.district} value={item.district}>
                  {item.district} — {item.name}
                </option>
              ))}
            </select>
          </div>
          <div className="field">
            <label>Month</label>
            <select value={month} onChange={(event) => patchSession({ month: event.target.value })}>
              {monthOptions.map((label, index) => (
                <option key={label} value={index + 1}>
                  {label}
                </option>
              ))}
            </select>
          </div>
          <div className="field">
            <label>Year</label>
            <input type="number" min="2001" max="2100" value={year} onChange={(event) => patchSession({ year: event.target.value })} />
          </div>
          <button className="primary" onClick={predict} disabled={loading || !district || !crime}>
            {loading ? "Predicting..." : "Predict Hotspot"}
          </button>
        </div>
      </Section>

      {error && <div className="message error">{error}</div>}
      {loading && (
        <div className="message">
          {modelSwitching
            ? "Updating forecast with the selected regression model..."
            : "Training regression models and generating future hotspot forecast..."}
        </div>
      )}

      {result && (
        <>
          <div className="grid stats" key={`forecast-${displayModel}-${result.predicted_crimes}-${result.risk_score}`}>
            <Stat label="Crime category" value={result.crime || crime} detail={CARD_DETAILS.forecastCrime} />
            <Stat label="District" value={`${result.district} — ${result.district_name}`} detail={CARD_DETAILS.forecastDistrict} />
            <Stat label="Predicted crimes" value={formatPredictedCrimes(result.predicted_crimes)} detail={CARD_DETAILS.forecastPredicted} />
            <Stat label="Active model" value={displayModel || "—"} detail={CARD_DETAILS.forecastActiveModel} />
          </div>
          <div className="grid stats">
            <Stat label="Risk score" value={result.risk_score} detail={CARD_DETAILS.forecastRiskScore} />
            <InfoCard className="stat" detail={CARD_DETAILS.forecastRiskLevel}>
              <span className="muted">Risk level</span>
              <div className="stat-custom-value">
                <RiskBadge level={result.risk_level} />
              </div>
            </InfoCard>
            <Stat label="Best model (R²)" value={`${bestModel || "—"} · ${bestModelR2.toFixed(4)}`} detail={CARD_DETAILS.forecastBestModel} />
            <Stat label="Active model R²" value={activeModelR2.toFixed(4)} detail={CARD_DETAILS.forecastActiveR2} />
          </div>
          {result.selectionReason && (
            <p className="muted table-note">{result.selectionReason}</p>
          )}
          <p className="muted table-note">
            Click any row in <strong>Regression Model Evaluation</strong> to switch the active model and refresh predictions, map, and trend chart.
          </p>

          <div className="grid two">
            <InfoCard className="risk-card" title="Forecast Summary" detail={CARD_DETAILS.forecastSummary}>
              <p className="muted table-note">
                Crime: {result.crime || crime}. Prediction target: {result.month_name} {result.year}. Active model: {displayModel}.
              </p>
              <div className="risk-summary">
                <div>
                  <span className="muted">Expected crime count</span>
                  <strong>{formatPredictedCrimes(result.predicted_crimes)}</strong>
                </div>
                <div>
                  <span className="muted">Risk score</span>
                  <strong>{result.risk_score}</strong>
                </div>
                <div>
                  <span className="muted">Risk level</span>
                  <RiskBadge level={result.risk_level} />
                </div>
              </div>
              <p className="muted table-note">
                Dynamic thresholds — LOW: ≤ {Math.round(result.riskThresholds?.low_max || 0)}, MEDIUM: ≤ {Math.round(result.riskThresholds?.medium_max || 0)}, HIGH: above medium band.
              </p>
            </InfoCard>

            <InfoCard className="compact-table-card" title="Regression Model Evaluation" detail={CARD_DETAILS.regressionEvaluation} interactive>
              <p className="muted table-note">★ = best R² · highlighted row = active model for predictions</p>
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Model</th>
                      <th>MAE</th>
                      <th>RMSE</th>
                      <th>R2</th>
                    </tr>
                  </thead>
                  <tbody>
                    {evaluationRows.map((row) => (
                      <tr
                        key={row.Model}
                        className={`model-select-row${row.isActive ? " active-model-row" : ""}${row.isBest ? " best-model-row" : ""}${row.isUnavailable ? " model-row-unavailable" : ""}${loading ? " model-select-row--busy" : ""}`}
                        onClick={() => !row.isUnavailable && selectModel(row.Model)}
                        title={row.isUnavailable ? "This model failed to train" : loading ? "Wait for the current forecast to finish" : "Use this model for predictions and trend chart"}
                      >
                        <td>{row.Model}{row.isBest ? " ★" : ""}{row.isActive ? " ✓" : ""}</td>
                        <td>{Number(row.MAE).toFixed(3)}</td>
                        <td>{Number(row.RMSE).toFixed(3)}</td>
                        <td>{Number(row.R2).toFixed(4)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </InfoCard>
          </div>

          <Section
            eyebrow="Trend"
            title="Actual Crime Trend vs Predicted Crime Trend"
            description={`Historical ${result.crime || crime} counts for district ${result.district_name} compared with predictions from ${displayModel}.`}
          >
            <InfoCard title="Actual vs Predicted Trend" detail={CARD_DETAILS.forecastTrend} interactive>
              <div className="chart chart-medium">
                <ChartView
                  key={`${displayModel}-${result.predicted_crimes}-${(result.trend?.predicted || []).join(",")}`}
                  type="line"
                  labels={result.trend?.labels || []}
                  valueDecimals={1}
                  datasets={[
                    { label: `Actual (${result.crime || crime})`, data: result.trend?.actual || [], borderColor: "#2563eb" },
                    { label: `Predicted (${displayModel})`, data: result.trend?.predicted || [], borderColor: "#0f766e" },
                  ]}
                  xLabel="Month"
                  yLabel="Crime count"
                />
              </div>
            </InfoCard>
          </Section>

          <Section
            eyebrow="Map"
            title="Future Hotspot Map"
            description="Green markers indicate low future risk, yellow markers medium risk, and red markers high future risk."
          >
            <InfoCard title="Future Hotspot Map" detail={CARD_DETAILS.futureHotspotHeatmap} interactive modalWide>
              <div className="legend-list map-legend">
                <span><i style={{ background: "#16a34a" }} /> LOW</span>
                <span><i style={{ background: "#eab308" }} /> MEDIUM</span>
                <span><i style={{ background: "#dc2626" }} /> HIGH</span>
              </div>
              <MapView key={`hotspot-map-${displayModel}`} points={result.mapPoints} heatmap={false} />
            </InfoCard>
          </Section>

          <Section
            eyebrow="Districts"
            title="All District Future Predictions"
            description="Regression forecast and risk classification for every police district in the uploaded dataset."
          >
            <InfoCard className="compact-table-card" title="All District Forecasts" detail={CARD_DETAILS.allDistrictPredictions} interactive>
              <DataTable
                rows={(result.allDistrictPredictions || []).map((row) => ({
                  District: row.district,
                  Name: row.district_name,
                  Month: row.month_name,
                  Year: row.year,
                  Predicted_Crimes: Math.round(row.predicted_crimes),
                  Risk_Score: row.risk_score,
                  Risk_Level: row.risk_level,
                }))}
              />
            </InfoCard>
          </Section>
        </>
      )}
    </div>
  );
}

function NavIcon({ type }) {
  if (type === "home") {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="M3 10.5L12 3l9 7.5" />
        <path d="M6 9.5V21h12V9.5" />
        <path d="M10 21v-5h4v5" />
      </svg>
    );
  }
  if (type === "overview") {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <rect x="4" y="4" width="6" height="6" rx="1.5" />
        <rect x="14" y="4" width="6" height="6" rx="1.5" />
        <rect x="4" y="14" width="6" height="6" rx="1.5" />
        <rect x="14" y="14" width="6" height="6" rx="1.5" />
      </svg>
    );
  }
  if (type === "analysis") {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <circle cx="11" cy="11" r="6.5" />
        <path d="M20 20l-4.2-4.2" />
        <path d="M8.5 11h5" />
        <path d="M11 8.5v5" />
      </svg>
    );
  }
  if (type === "ml") {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <rect x="4" y="4" width="16" height="16" rx="2.5" />
        <path d="M8 8h8M8 12h8M8 16h5" />
        <circle cx="17.5" cy="16.5" r="2.2" />
      </svg>
    );
  }
  if (type === "future") {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="M4 18l4-5 3 2 5-7 4 2" />
        <path d="M4 20h16" />
        <path d="M17 4l3 1-1 3" />
      </svg>
    );
  }
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <circle cx="12" cy="12" r="8" />
      <path d="M12 10v6" />
      <circle cx="12" cy="7.5" r="0.8" fill="currentColor" stroke="none" />
    </svg>
  );
}

function App() {
  const [page, setPage] = useState("Home");
  const [data, setData] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [message, setMessage] = useState("");
  const [fileName, setFileName] = useState("");
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [mlSession, setMlSession] = useState(INITIAL_ML_SESSION);
  const [futureSession, setFutureSession] = useState(INITIAL_FUTURE_SESSION);

  const datasetSummary = useMemo(() => {
    if (!data) return null;
    return `${formatNumber(data.rows)} clean rows · ${formatNumber(data.summary?.crimeTypes || 0)} crime types · ${formatNumber(data.summary?.districts || 0)} districts`;
  }, [data]);

  const pageTitle = useMemo(() => {
    if (page === "Overview") return "Data Overview";
    if (page === "Analysis") return "Geospatial Analysis";
    if (page === "Machine Learning") return "Machine Learning";
    if (page === "Future Crime Prediction") return "Future Crime Prediction";
    if (page === "About Project") return "About Project";
    return "Project Home";
  }, [page]);

  const navItems = [
    { label: "Home", icon: "home", tone: "home" },
    { label: "Overview", icon: "overview", tone: "overview" },
    { label: "Analysis", icon: "analysis", tone: "analysis" },
    { label: "Machine Learning", icon: "ml", tone: "ml" },
    { label: "Future Crime Prediction", icon: "future", tone: "future" },
    { label: "About Project", icon: "about", tone: "about" },
  ];

  function uploadFile(fileOrEvent) {
    const file = fileOrEvent?.target?.files?.[0] || fileOrEvent;
    if (!file) return;
    const form = new FormData();
    form.append("file", file);
    setUploading(true);
    setMessage("");
    setFileName(file.name);
    fetch("/api/upload", { method: "POST", body: form })
      .then((res) => res.json().then((body) => (res.ok ? body : Promise.reject(body))))
      .then((body) => {
        setData(body);
        setMlSession(INITIAL_ML_SESSION);
        setFutureSession(INITIAL_FUTURE_SESSION);
        setPage("Overview");
        setMessage(`Cleaned ${formatNumber(body.cleaning?.originalRows || body.rows)} raw rows into ${formatNumber(body.rows)} usable records.`);
      })
      .catch((err) => setMessage(err.error || "Upload failed."))
      .finally(() => setUploading(false));

    if (fileOrEvent?.target) {
      fileOrEvent.target.value = "";
    }
  }

  return (
    <div className={`app${sidebarCollapsed ? " app-sidebar-collapsed" : ""}`}>
      <aside className="sidebar">
        <button
          type="button"
          className="sidebar-toggle"
          onClick={() => setSidebarCollapsed((v) => !v)}
          aria-label={sidebarCollapsed ? "Expand navigation" : "Collapse navigation"}
          title={sidebarCollapsed ? "Expand navigation" : "Collapse navigation"}
        >
          {sidebarCollapsed ? "»" : "«"}
        </button>
        <div className="brand">
          <div
            className="brand-mark"
            role="button"
            tabIndex={0}
            title="Go to homepage"
            onClick={() => setPage("Home")}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") setPage("Home");
            }}
          />
          <h1>Cyber Crime Dashboard</h1>
        </div>
        <nav className="nav">
          {navItems.map((item) => (
            <button
              className={page === item.label ? "active" : ""}
              onClick={() => setPage(item.label)}
              key={item.label}
              title={sidebarCollapsed ? item.label : undefined}
              aria-label={item.label}
            >
              <span className={`nav-icon nav-icon-${item.tone}`} aria-hidden="true">
                <NavIcon type={item.icon} />
              </span>
              <span className="nav-label">{item.label}</span>
            </button>
          ))}
        </nav>
        {datasetSummary && (
          <div className="dataset-badge">
            <span className="muted">Loaded dataset</span>
            <strong>{datasetSummary}</strong>
          </div>
        )}
      </aside>

      <section className="content">
        <header className="topbar">
          <div>
            <h2>{pageTitle}</h2>
            {message && <p className={message.includes("failed") || message.includes("Missing") ? "muted error-text" : "muted"}>{message}</p>}
          </div>
          <label className="upload upload-topbar">
            <div className="upload-topbar-label">
              <div className="upload-topbar-main">{uploading ? "Cleaning..." : "Upload dataset"}</div>
              <div className="upload-topbar-sub">Upload raw dataset</div>
            </div>
            <input type="file" accept=".csv,.xlsx" onChange={uploadFile} />
          </label>
        </header>

        <main className="main">
          {page === "Home" && (
            <Home
              onUpload={uploadFile}
              uploading={uploading}
              message={message}
              hasData={Boolean(data)}
              fileName={fileName}
            />
          )}
          {page === "Overview" && <Overview data={data} />}
          {page === "Analysis" && <Analysis data={data} />}
          {page === "Machine Learning" && (
            <MachineLearning data={data} session={mlSession} setSession={setMlSession} />
          )}
          {page === "Future Crime Prediction" && (
            <FutureCrimePrediction data={data} session={futureSession} setSession={setFutureSession} />
          )}
          {page === "About Project" && <AboutProject />}
        </main>
      </section>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
