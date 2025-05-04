const chartTitle = document.getElementById('chartTitle');
const cryptoButtons = document.querySelectorAll('.crypto-tabs button[data-coin]');
const settingsToggle = document.getElementById('settingsToggle');
const settingsPanel = document.getElementById('settingsPanel');
const changeColorCheckbox = document.getElementById('changeColorCheckbox');
const deactivateLinesCheckbox = document.getElementById('deactivateLinesCheckbox');
const candleDetails = document.getElementById('candleDetails');

let chart;
let currentSymbol = 'BTCUSDT';

let currentColors = {
  up: '#00b894',
  down: '#d63031',
  unchanged: '#888'
};

async function fetchCandleData(symbol) {
  const response = await fetch(`https://api.binance.com/api/v3/klines?symbol=${symbol}&interval=1h&limit=24`);
  const data = await response.json();
  return data.map(candle => ({
    x: candle[0],
    o: parseFloat(candle[1]),
    h: parseFloat(candle[2]),
    l: parseFloat(candle[3]),
    c: parseFloat(candle[4])
  }));
}

async function renderChart(symbol = 'BTCUSDT') {
  const data = await fetchCandleData(symbol);
  const ctx = document.getElementById('chartCanvas').getContext('2d');

  if (chart) chart.destroy();

  chart = new Chart(ctx, {
    type: 'candlestick',
    data: {
      datasets: [{
        label: `${symbol} 24H`,
        data: data,
        color: {
          up: currentColors.up,
          down: currentColors.down,
          unchanged: currentColors.unchanged
        }
      }]
    },
    options: {
      layout: { padding: 20 },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: function (context) {
              const candle = context.raw;
              return [
                `Open: ${candle.o}`,
                `High: ${candle.h}`,
                `Low: ${candle.l}`,
                `Close: ${candle.c}`
              ];
            }
          }
        },
        zoom: {
          pan: { enabled: true, mode: 'x' },
          zoom: {
            wheel: { enabled: true },
            pinch: { enabled: true },
            mode: 'x'
          }
        }
      },
      scales: {
        x: {
          type: 'time',
          time: { unit: 'hour' },
          grid: {
            display: !deactivateLinesCheckbox.checked,
            color: '#dfe6e9'
          },
          ticks: {
            color: '#2d3436',
            font: { weight: 'bold' }
          }
        },
        y: {
          beginAtZero: false,
          grid: {
            display: !deactivateLinesCheckbox.checked,
            color: '#dfe6e9'
          },
          ticks: {
            color: '#2d3436',
            font: { weight: 'bold' }
          }
        }
      },
      onHover: (event, elements) => {
        if (elements.length) {
          const candle = elements[0].element.$context.raw;
          candleDetails.innerHTML = `
            <strong>Time:</strong> ${new Date(candle.x).toLocaleString()}<br>
            <strong>Open:</strong> ${candle.o}<br>
            <strong>High:</strong> ${candle.h}<br>
            <strong>Low:</strong> ${candle.l}<br>
            <strong>Close:</strong> ${candle.c}
          `;
        }
      }
    }
  });
}

document.addEventListener('DOMContentLoaded', () => {
  const canvas = document.createElement('canvas');
  canvas.id = 'chartCanvas';
  canvas.width = 800;
  canvas.height = 400;
  document.querySelector('.chart-wrapper').appendChild(canvas);
  renderChart(currentSymbol);

  cryptoButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const coin = btn.dataset.coin;
      const symbol = mapCoinToSymbol(coin);
      currentSymbol = symbol;
      chartTitle.textContent = `${coin} / USD`;
      renderChart(symbol);
      fetchPrices(symbol);
    });
  });

  settingsToggle.addEventListener('click', () => {
    settingsPanel.style.display = settingsPanel.style.display === 'block' ? 'none' : 'block';
  });

  changeColorCheckbox.addEventListener('change', () => {
    currentColors.up = changeColorCheckbox.checked ? '#0984e3' : '#00b894';
    currentColors.down = changeColorCheckbox.checked ? '#fdcb6e' : '#d63031';
    currentColors.unchanged = '#888';
    renderChart(currentSymbol);
  });

  deactivateLinesCheckbox.addEventListener('change', () => {
    if (!chart) return;
    const showGrid = !deactivateLinesCheckbox.checked;
    chart.options.scales.x.grid.display = showGrid;
    chart.options.scales.y.grid.display = showGrid;
    chart.update();
  });
});

function mapCoinToSymbol(coinName) {
  switch (coinName) {
    case 'Bitcoin': return 'BTCUSDT';
    case 'Ethereum': return 'ETHUSDT';
    case 'TonCoin': return 'TONUSDT';
    case 'Solana': return 'SOLUSDT';
    default: return 'BTCUSDT';
  }
}

async function fetchPrices(symbol = 'BTCUSDT') {
  const currentPriceUrl = `https://api.binance.com/api/v3/ticker/price?symbol=${symbol}`;
  const priceChangeUrl = `https://api.binance.com/api/v3/ticker/24hr?symbol=${symbol}`;
  
  try {
    const currentPriceResponse = await fetch(currentPriceUrl);
    const currentPriceData = await currentPriceResponse.json();
    let currentPrice = parseFloat(currentPriceData.price);
    currentPrice = currentPrice.toFixed(3);
    
    const priceChangeResponse = await fetch(priceChangeUrl);
    const priceChangeData = await priceChangeResponse.json();
    let price1HourChange = parseFloat(priceChangeData.priceChangePercent);
    let price24HourChange = parseFloat(priceChangeData.priceChangePercent);

    const price1Hour = (currentPrice * (1 + price1HourChange / 100)).toFixed(3);
    const price24Hour = (currentPrice * (1 + price24HourChange / 100)).toFixed(3);

    document.getElementById('currentPrice').innerHTML = `${currentPrice}$`;
    document.getElementById('price1Hour').innerHTML = `${price1Hour}$ (${price1HourChange > 0 ? '+' : ''}${price1HourChange.toFixed(2)}%)`;
    document.getElementById('price24Hour').innerHTML = `${price24Hour}$ (${price24HourChange > 0 ? '+' : ''}${price24HourChange.toFixed(2)}%)`;

    const price1HourElement = document.getElementById('price1Hour');
    const price24HourElement = document.getElementById('price24Hour');
    
    if (price1HourChange < 0) {
      price1HourElement.classList.add('red');
      price1HourElement.classList.remove('green');
    } else {
      price1HourElement.classList.add('green');
      price1HourElement.classList.remove('red');
    }

    if (price24HourChange < 0) {
      price24HourElement.classList.add('red');
      price24HourElement.classList.remove('green');
    } else {
      price24HourElement.classList.add('green');
      price24HourElement.classList.remove('red');
    }

  } catch (error) {
    console.error("Error fetching data from Binance API:", error);
  }
}

fetchPrices();

document.getElementById('forecastCheckbox').addEventListener('change', async function () {
  const forecastSection = document.getElementById('forecastSection');
  const canvas = document.getElementById('forecastChart');

  if (this.checked) {
    forecastSection.style.display = 'flex'; 

    try {
      const response = await fetch('http://localhost:8000/predictions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ steps: 24 })
      });

      const data = await response.json();
      console.log("Response data:", data);

      if (response.ok && Array.isArray(data)) {
        updateForecastChart(data);
      } else {
        console.error("Error fetching predictions", data);
      }

    } catch (error) {
      console.error("Fetch error:", error);
    }

  } else {
    forecastSection.style.display = 'none'; 
    if (window.forecastChartInstance) {
      window.forecastChartInstance.destroy();
      window.forecastChartInstance = null;
    }
  }
});

function updateForecastChart(predictions) {
  const forecastData = predictions.map(p => ({
    x: new Date(p.timestamp * 1000),
    y: p.close
  }));

  const ctx = document.getElementById('forecastChart').getContext('2d');

  if (window.forecastChartInstance) {
    window.forecastChartInstance.destroy();
  }

  window.forecastChartInstance = new Chart(ctx, {
    type: 'line',
    data: {
      datasets: [{
        label: 'Forecast (Line Chart)',
        data: forecastData,
        borderColor: '#2980b9',
        borderWidth: 2,
        fill: false,
        tension: 0.3,
        pointRadius: 3
      }]
    },
    options: {
      responsive: true,
      scales: {
        x: {
          type: 'time',
          time: {
            tooltipFormat: 'HH:mm',
            displayFormats: {
              hour: 'HH:mm'
            }
          },
          title: {
            display: true,
            text: 'Time'
          }
        },
        y: {
          title: {
            display: true,
            text: 'Price'
          }
        }
      },
      plugins: {
        title: {
          display: true,
          text: 'Forecast Chart'
        }
      }
    }
  });
}

document.addEventListener('DOMContentLoaded', () => {
  const forecastSection = document.getElementById('forecastSection');
  const canvas = document.getElementById('forecastHistoryChart');
  
  async function fetchHistoryData() {
    const response = await fetch('http://localhost:8000/history');
    const data = await response.json();

    const timestamps = data.results.map(item => new Date(item.timestamp * 1000).toLocaleString());
    const predicted = data.results.map(item => item.predicted);
    const real = data.results.map(item => item.real);

    const ctx = canvas.getContext('2d');
    new Chart(ctx, {
      type: 'line',
      data: {
        labels: timestamps,
        datasets: [
          {
            label: 'Predicted',
            data: predicted,
            borderColor: 'green',
            backgroundColor: 'rgba(0, 255, 0, 0.1)',
            fill: false,
            tension: 0.2,
          },
          {
            label: 'Real',
            data: real,
            borderColor: 'red',
            backgroundColor: 'rgba(255, 0, 0, 0.1)',
            fill: false,
            tension: 0.2,
          }
        ]
      },
      options: {
        responsive: true,
        plugins: {
          title: {
            display: true,
            text: 'Predicted vs Real Price History'
          }
        },
        scales: {
          x: {
            title: {
              display: true,
              text: 'Time'
            }
          },
          y: {
            title: {
              display: true,
              text: 'Price'
            },
            beginAtZero: false
          }
        }
      }
    });
  }

  fetchHistoryData();
});