import * as tf from "@tensorflow/tfjs";
import * as tfvis from "@tensorflow/tfjs-vis";
import * as normalization from "./normalization";
import * as ui from "./ui";

import { BostonHousingDataset, featureDescriptions } from "./data";
import { norm } from "@tensorflow/tfjs";

const NUM_EPOCHS = 200;
const BATCH_SIZE = 40;
const LEARNING_RATE = 0.01;

const bostonData = new BostonHousingDataset();
const tensors = {};
//converting csv into tensors

export function arraysToTensors() {
  tensors.rawTrainFeatures = tf.tensor2d(bostonData.trainFeatures);
  tensors.trainTarget = tf.tensor2d(bostonData.trainTarget);
  tensors.rawTestFeatures = tf.tensor2d(bostonData.testFeatures);
  tensors.testTarget = tf.tensor2d(bostonData.testTarget);

  let { dataMean, dataStd } = normalization.determineMeanAndStdDev(
    tensors.rawTrainFeatures
  );

  tensors.trainFeatures = normalization.normalizeTheTensor(
    tensors.rawTrainFeatures,
    dataMean,
    dataStd
  );

  tensors.testFeatures = normalization.normalizeTheTensor(
    tensors.rawTestFeatures,
    dataMean,
    dataStd
  );
}

export const linearRegressionModel = () => {
  const model = tf.sequential();
  model.add(
    tf.layers.dense({
      inputShape: [bostonData.numFeatures], //get the length
      units: 1
    })
  );

  model.summary();

  return model;
};

if (document.readyState !== "loading") {
  (async () => {
    await bostonData.loadData();
    arraysToTensors();
    ui.updateStatus(
      "Data is now available as tensors.\nClick a train button to begin."
    );

    ui.updateBaselineStatus("Estimating baseline loss");
    ComputeBaseline();
    await ui.setup();
  })();
} else {
  document.addEventListener("DOMContentLoaded", async function () {
    await bostonData.loadData();
    arraysToTensors();
    ui.updateStatus(
      "Data is now available as tensors.\nClick a train button to begin."
    );

    ui.updateBaselineStatus("Estimating baseline loss");
    ComputeBaseline();
    await ui.setup();
  });
}

//computing Baseline and all
export const ComputeBaseline = () => {
  const avgPrice = tf.mean(tensors.trainTarget);
  console.log("avg price is " + avgPrice.dataSync()[0].toFixed(2));
  const baseline = tensors.testTarget.sub(avgPrice).square().mean();
  console.log(
    `Baseline Loss (MeanSquaredError) is ${baseline.dataSync()[0].toFixed(2)}`
  );
  const baselineMsg = `Baseline loss (MeanSquaredError) is ${baseline
    .dataSync()[0]
    .toFixed(2)}`;

  ui.updateBaselineStatus(baselineMsg);
};

export function describeKernelElements(kernel) {
  tf.util.assert(
    kernel.length == 12,
    `kernel must be a array of length 12, got ${kernel.length}`
  );
  const outList = [];
  for (let idx = 0; idx < kernel.length; idx++) {
    outList.push({ description: featureDescriptions[idx], value: kernel[idx] });
  }
  return outList;
}
export async function run(model, modelName, weightsIllustrations) {
  model.compile({
    optimizer: tf.train.sgd(LEARNING_RATE),
    loss: "meanSquaredError"
  });

  let trainLogs = [];
  const container = document.querySelector(`#${modelName} .chart`);

  ui.updateStatus("Starting training process");
  await model.fit(tensors.trainFeatures, tensors.trainTarget, {
    batchSize: BATCH_SIZE,
    epochs: NUM_EPOCHS,
    validationSplit: 0.2,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        await ui.updateModelStatus(
          `Epoch ${epoch + 1} of ${NUM_EPOCHS} completed`,
          modelName
        );
        trainLogs.push(logs);
        tfvis.show.history(container, trainLogs, ["loss", "val_loss"]);

        if (weightsIllustrations) {
          model.layers[0]
            .getWeights()[0]
            .data()
            .then((kernelAsArr) => {
              const weightsList = describeKernelElements(kernelAsArr);
              ui.updateWeightDescription(weightsList);
            });
        }
      }
    }
  });

  ui.updateStatus("Running on test data...");
  const result = model.evaluate(tensors.testFeatures, tensors.testTarget, {
    batchSize: BATCH_SIZE
  });

  const testLoss = result.dataSync()[0];

  const trainLoss = trainLogs[trainLogs.length - 1].loss;
  const valLoss = trainLogs[trainLogs.length - 1].val_loss;

  await ui.updateModelStatus(
    `Final train-set loss: ${trainLoss.toFixed(4)}\n` +
      `Final validation-set loss: ${valLoss.toFixed(4)}\n` +
      `Test-set loss: ${testLoss.toFixed(4)}`,
    modelName
  );
}
