const model = tf.sequential()

async function Entrenar() {
    const repeticiones = parseInt(document.getElementById('repeticiones').value);
     const epochs = repeticiones;

    model.add(tf.layers.dense({units: 1, inputShape: [1]}));


    model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});
 
     //entrenando con formula 3x + 2
    const xs = tf.tensor2d([-1, 0, 1, 2, 3, 4, 5, 6], [8, 1]);
    const ys = tf.tensor2d([-1, 2, 5, 8, 11, 14, 17, 20], [8, 1]);


    const history = await model.fit(xs, ys, {
        epochs: epochs,
        callbacks: {
          onEpochEnd: (epoch, logs) => {
             console.log(logs);
             console.log("/n");
             console.log(`Epoch ${epoch+1} - Loss: ${logs.loss.toFixed(4)},`);
          }
        }
      });

      // Imprimir la pérdida final
      console.log(`Final Loss: ${history.history.loss[epochs-1].toFixed(4)}`);

      alert("terminó de entrenar");
}


async function Predecir() {
    const prediccionValor = parseInt(document.getElementById('valorPredecir').value);

    document.getElementById('Resultado').innerText =
    model.predict(tf.tensor2d([prediccionValor], [1, 1])).dataSync();
}