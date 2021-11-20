import { linearRegressionModel, run } from "./index";

const statusElement = document.getElementById("status");
export function updateStatus(message) {
  statusElement.innerText = message;
}

const baselineStatusElement = document.getElementById("baselineStatus");
export const updateBaselineStatus = (message) => {
  baselineStatusElement.innerText = message;
};

export const updateModelStatus = (message, modelName) => {
  const stateElement = document.querySelector(`#${modelName} .status`);
  stateElement.innerText = message;
};

const NUM_TOP_WEIGHT_TO_DISPLAY = 5;

export const updateWeightDescription = (weightsList) => {
  const inspectionHeadlineElement = document.getElementById(
    "inspectionHeadline"
  );

  inspectionHeadlineElement.innerText = `Top ${NUM_TOP_WEIGHT_TO_DISPLAY} weights by magnitude`;

  weightsList.sort((a, b) => Math.abs(b.value) - Math.abs(a.value));

  var table = document.getElementById("myTable");

  table.innerHTML = "";

  weightsList.forEach((el, i) => {
    if (i < NUM_TOP_WEIGHT_TO_DISPLAY) {
      let row = table.insertRow(-1);
      let cell1 = row.insertCell(0);
      let cell2 = row.insertCell(1);

      if (el.value < 0) {
        cell2.setAttribute("class", "negativeWeight");
      } else {
        cell2.setAttribute("class", "positiveWeight");
      }

      cell1.innerHTML = el.description;
      cell2.innerHTML = el.value.toFixed(4);
    }
  });
};

export const setup = async () => {
  const trainSimpleLeaninearRegression = document.getElementById("simple-mlr");

  trainSimpleLeaninearRegression.addEventListener("click", async () => {
    const model = linearRegressionModel();
    await run(model, "linear", true);
  });
};
