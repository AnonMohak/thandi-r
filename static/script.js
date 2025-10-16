/* Fetch chart data and render charts; handle predictions */

let barChartRef = null;
let deptBarChartRef = null;
let scatterChartRef = null;

async function loadChartData() {
  try {
    const res = await axios.get("/chart-data");
    const payload = res.data;

    const byMarket = payload.byMarket || { labels: [], data: [] };
    const byDepartment = payload.byDepartment || { labels: [], data: [] };
    const stockVsDelay = payload.stockVsDelay || { x: [], y: [] };

    renderBarChart(byMarket.labels, byMarket.data);
    renderDeptBarChart(byDepartment.labels, byDepartment.data);
    renderScatterChart(stockVsDelay.x, stockVsDelay.y);
  } catch (err) {
    console.error("Failed to load chart data:", err);
  }
}

function renderBarChart(labels, data) {
  const ctx = document.getElementById("barChart").getContext("2d");
  if (barChartRef) {
    barChartRef.destroy();
  }
  barChartRef = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: "Avg Delivery Days",
          data,
          backgroundColor: "rgba(91, 124, 250, 0.6)",
          borderColor: "rgba(91, 124, 250, 1)",
          borderWidth: 1.2,
          borderRadius: 8,
        },
      ],
    },
    options: {
      responsive: true,
      animation: {
        duration: 650
      },
      scales: {
        y: {
          beginAtZero: true,
          title: { display: true, text: "Days" }
        }
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => ` ${ctx.parsed.y} days`,
          },
        },
      },
    },
  });
}

function renderDeptBarChart(labels, data) {
  const ctx = document.getElementById("deptBarChart").getContext("2d");
  if (deptBarChartRef) {
    deptBarChartRef.destroy();
  }
  deptBarChartRef = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: "Avg Delivery Days",
          data,
          backgroundColor: "rgba(106, 138, 255, 0.6)",
          borderColor: "rgba(106, 138, 255, 1)",
          borderWidth: 1.2,
          borderRadius: 8,
        },
      ],
    },
    options: {
      responsive: true,
      animation: { duration: 650 },
      scales: {
        y: { beginAtZero: true, title: { display: true, text: "Days" } },
      },
      plugins: { legend: { display: false } },
    },
  });
}

function renderScatterChart(x, y) {
  const ctx = document.getElementById("scatterChart").getContext("2d");
  if (scatterChartRef) scatterChartRef.destroy();
  const points = x.map((xi, i) => ({ x: xi, y: y[i] }));
  scatterChartRef = new Chart(ctx, {
    type: "scatter",
    data: {
      datasets: [
        {
          label: "Stock vs Delay",
          data: points,
          backgroundColor: "rgba(91, 124, 250, 0.6)",
          borderColor: "rgba(91, 124, 250, 1)",
        },
      ],
    },
    options: {
      responsive: true,
      animation: { duration: 650 },
      scales: {
        x: { title: { display: true, text: "Stock Level" } },
        y: { title: { display: true, text: "Days for shipping (real)" } },
      },
      plugins: { legend: { display: false } },
    },
  });
}

function setupPredictForm() {
  const form = document.getElementById("predictForm");
  const resultCard = document.getElementById("predictionResult");
  const predictedValueEl = document.getElementById("predictedValue");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    // Collect values
    const formData = new FormData(form);
    const payload = {};
    for (const [key, value] of formData.entries()) {
      payload[key] = value;
    }

    // Coerce numbers where appropriate
    const toNumber = (val) => (val === "" || val === null ? null : Number(val));
    payload["Days for shipment (scheduled)"] = toNumber(payload["Days for shipment (scheduled)"]);
    payload["Late_delivery_risk"] = toNumber(payload["Late_delivery_risk"]);
    payload["Sales"] = toNumber(payload["Sales"]);
    payload["Benefit per order"] = toNumber(payload["Benefit per order"]);
    payload["Order Profit Per Order"] = toNumber(payload["Order Profit Per Order"]);

    try {
      const res = await axios.post("/predict", payload);
      const data = res.data;
      if (data && typeof data.predicted_days_for_shipping !== "undefined") {
        predictedValueEl.textContent = `${data.predicted_days_for_shipping} days`;
        resultCard.classList.remove("hidden");
      } else if (data.error) {
        predictedValueEl.textContent = `Error: ${data.error}`;
        resultCard.classList.remove("hidden");
      }
    } catch (err) {
      predictedValueEl.textContent = `Error: ${err?.response?.data?.error || err.message}`;
      resultCard.classList.remove("hidden");
    }
  });
}

document.addEventListener("DOMContentLoaded", () => {
  loadChartData();
  setupPredictForm();
});
