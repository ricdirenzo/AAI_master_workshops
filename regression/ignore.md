## Struttura e funzionamento

Una FFNN è un tipo di rete neurale - la più semplice - in cui le info fluiscono in una sola direzione (solo in avanti appunto) dagli input all'output attraverso uno o più livelli nascosti. È costituita da 3 livelli:

1. **un livello di input** un vettore delle features
2. **uno o più livello/i nascosto/i** costituito/i da un certo numero di neuroni
3. **un livello di output** un vettore target

Ciascun neurone prende in input ciò che arriva dal livello precedente come somma pesata e vi applica una funzione di attivazione e trasmette il risultato al livello successivo (output). Ogni nodo è collegato a tutti gli altri formando una rete completamente connessa. Le FFNN vengono usate per compiti di classificazione/regressione.

**input → livello nascosto**

Sui neuroni - del livello nascosto - viene calcolata una somma delle features pesata a cui viene aggiunto un termine di bias: 

$$
z_i=\sum_{j=1}^n{x_{j}w_{ij}+b_i}
$$

Quelli che ne risultano sono i valori pre-attivati del livello nascosto. 

**livello nascosto**

Ai valori pre-attivati viene applicata una **funzione di attivazione** - il cui compito è quello di introdurre una non linearità permettendo alla rete di apprendere le relazioni nei dati - in questo caso una ReLU randomizzata: una variante della ReLU che introduce una pendenza casuale nella parte negativa per evitare il problema di neuroni morti (quando troppi valori diventano zero)

$$
a_i = \begin{cases}
z_i & \text{if } z_i \geq 0 \\
\alpha_i z_i & \text{if } z_i < 0
\end{cases}
$$

questo termine di pendenza viene campionato da una dist. uniforme.

**livello nascosto → livello di output**

Sui neuroni - del livello di output - viene calcolata una somma delle attivazioni pesata a cui viene aggiunto un termine di bias:

$$
z_j=\sum_{i=1}^h{a_{i}w_{ji}+b_j}
$$

Quelli che ne risultano sono i valori pre-attivati del livello di output, che corrispondono alle predizioni finali. 

## Addestramento

L'addestramento di una FFNN consiste in due fasi:

1. **Feed-Forward**
   i dati passano dagli input, attraverso i vari livelli fino all'output (predizione). Qui viene applicata:
   
   - una trasformazione lineare (input → livello nascosto)
   - normalizzazione batch
   - funzione di attivazione
   - una trasformazione lineare (livello nascosto → output) <br /><br />

3. **Back propagation**
   una volta ottenuta la predizione, vengono:

   - calcolato l'errore quadratico medio (MSE)
   - azzerati "puliti" i gradienti (per evitare che si sommino a quelli dell'iterazione precedente)
   - calcolati i gradienti dell'errore rispetto ai pesi della rete, questa volta a partire dall'output e all'indietro propagandosi fino all'input
   - aggiornati i pesi.
