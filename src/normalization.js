export const determineMeanAndStdDev = (data) => {
  const dataMean = data.mean(0);

  //why 0 because we need mean for each column, i.e. for each feature
  //and since it is 2d tensor we speaking of, so this is better if we keep it that way
  //and if put 1 as argument then for each housing data, it will give us mean by using alll the columns in one row
  //absolutely not wwhat we want here right??

  const diffFromMean = data.sub(dataMean);
  //something internal will happen
  //so see data has 333 rows for training set and datamean is just a single row with mean for each feature
  //now internally tensorflow will use datamean exact number of times the row in data

  const squaredDiffFromMean = diffFromMean.square();
  const variance = squaredDiffFromMean.mean(0);
  const dataStd = variance.sqrt();
  return { dataMean, dataStd };
};

export const normalizeTheTensor = (data, dataMean, dataStd) => {
  return data.sub(dataMean).div(dataStd);
};
