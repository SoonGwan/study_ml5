// 1. 데이터 로드, 여기선 목업 데이터로 ㅎㅎ

const data = [
  { r: 255, g: 0, b: 0, color: 'red-ish' },
  { r: 254, g: 0, b: 0, color: 'red-ish' },
  { r: 253, g: 0, b: 0, color: 'red-ish' },
  { r: 0, g: 255, b: 0, color: 'green-ish' },
  { r: 0, g: 254, b: 0, color: 'green-ish' },
  { r: 0, g: 253, b: 0, color: 'green-ish' },
  { r: 0, g: 0, b: 255, color: 'blue-ish' },
  { r: 0, g: 0, b: 254, color: 'blue-ish' },
  { r: 0, g: 0, b: 253, color: 'blue-ish' },
];

// 2. 뉴럴 넷웍 옵션 건들이자
const options = {
  task: 'classification',
  debug: true,
};

//3. 뉴럴 넷웍 초기화
const nn = ml5.neuralNetwork(options);

// 4. data를 nn에 추가
data.forEach((item) => {
  const { r, g, b, color } = item;
  const inputs = {
    r,
    g,
    b,
  };
  const output = {
    color,
  };

  nn.addData(inputs, output);
});

//5. data 정규화
nn.normalizeData();

//6. train neural neuralNetwork

const trainingOptions = {
  epochs: 32,
  batchSize: 12,
};

nn.train(trainingOptions, finishedTraining);

//7. train model use
function finishedTraining() {
  classify();
}
//8. classification
function classify() {
  const inputs = {
    r: 255,
    g: 0,
    b: 0,
  };
  nn.classify(inputs, handleResults);
}
//9. define handle results for 8. classification
function handleResults(error, result) {
  if (error) {
    console.error(error);
    return;
  }
  console.log(result);
}
