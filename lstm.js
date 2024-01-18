const tf = require('@tensorflow/tfjs');
const plotly = require('plotly')('Nikodem', 'X4UvmKRbeCCwYEmXvFWi');
const assert = require('assert');

// Normalizacja Min-Max Scaling
const normalizeData = (data) => {
  const min = Math.min(...data.flat());
  const max = Math.max(...data.flat());
  return data.map(seq => seq.map(value => (value - min) / (max - min)));
};

const transformInputData = (data) => {
  const flatData = data.flat();
  return tf.tensor3d(flatData, [data.length, data[0].length, 1]);
};

const createLSTMModel = () => {
  const model = tf.sequential();
  model.add(
    tf.layers.lstm({
      units: 100,
      inputShape: [null, 1],
      recurrentInitializer: 'orthogonal',
      returnSequences: true,
    })
  );
  model.add(
    tf.layers.lstm({
      units: 50,
      recurrentInitializer: 'orthogonal',
      returnSequences: true,
    })
  );
  model.add(
    tf.layers.timeDistributed({
      layer: tf.layers.dense({ units: 1 })
    })
  );
  model.add(tf.layers.dropout({ rate: 0.2 }));
  model.compile({ optimizer: 'rmsprop', loss: 'meanSquaredError' });
  return model;
};

// Zwiększone dane treningowe i walidacyjne
const trainingData = [
  [1, 2, 3, 4, 5, 6, 7],
  [2, 3, 4, 5, 6, 7, 8],
  [3, 4, 5, 6, 7, 8, 9],
  [4, 5, 6, 7, 8, 9, 10],
  [5, 6, 7, 8, 9, 10, 11],
  [6, 7, 8, 9, 10, 11, 12],
  [7, 8, 9, 10, 11, 12, 13],
  // Dodatkowe dane treningowe
  // ...
];

const validationData = [
  [8, 9, 10, 11, 12, 13, 14],
  [9, 10, 11, 12, 13, 14, 15],
  [10, 11, 12, 13, 14, 15, 16],
  // Dodatkowe dane walidacyjne
  // ...
];

// Normalizacja danych treningowych i walidacyjnych
const normalizedTrainingData = normalizeData(trainingData);
const normalizedValidationData = normalizeData(validationData);

const transformedData = transformInputData(normalizedTrainingData);
const transformedValidationData = transformInputData(normalizedValidationData);

const earlyStopping = tf.callbacks.earlyStopping({ monitor: 'val_loss', patience: 10 });

const model = createLSTMModel();

// Testy jednostkowe
function runUnitTests() {
  testTransformInputData();
  testCreateLSTMModel();
}

// Test funkcji transformInputData
function testTransformInputData() {
  const testData = [
    [1, 2, 3],
    [4, 5, 6]
  ];

  const transformedData = transformInputData(testData);
  assert.deepStrictEqual(transformedData.shape, [2, 3, 1], 'transformInputData shape test failed');
}

// Test funkcji createLSTMModel
function testCreateLSTMModel() {
  const model = createLSTMModel();
  assert.ok(model instanceof tf.Sequential, 'createLSTMModel test failed');
}

// Uruchomienie testów
runUnitTests();

// Proces trenowania
model.fit(transformedData, transformedData, {
  epochs: 150,
  validationData: [transformedValidationData, transformedValidationData],
  callbacks: [earlyStopping],
}).then((history) => {
  console.log('Training complete:', history);

  // Wizualizacja wyników
  const predictedData = model.predict(transformedValidationData).arraySync();
  plotResults(normalizedValidationData, predictedData);
}).catch((error) => {
  console.error('Error during training:', error);
});

function plotResults(actualData, predictedData) {
  // Przygotowanie danych do wykresu
  const actualTrace = {
    x: [...Array(actualData.length).keys()],
    y: actualData.map(seq => seq[seq.length - 1]),
    type: 'scatter',
    name: 'Actual',
  };

  const predictedTrace = {
    x: [...Array(predictedData.length).keys()],
    y: predictedData.map(seq => seq[seq.length - 1][0]),
    type: 'scatter',
    name: 'Predicted',
  };

  const layout = {
    title: 'Actual vs Predicted',
    xaxis: { title: 'Sequence Index' },
    yaxis: { title: 'Value' },
  };

  const chartData = [actualTrace, predictedTrace];
  const chartOptions = { layout: layout, filename: 'lstm-plot', fileopt: 'overwrite' };

  // Wygeneruj i wyślij wykres do Plotly
  plotly.plot(chartData, chartOptions, function (err, msg) {
    if (err) return console.error(err);
    console.log('Plotly chart URL:', msg.url);
  });
}
