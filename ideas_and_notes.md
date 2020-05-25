# Ideas and Notes

## 2D Embeddings
* Double-size vector R-GCN
  * Layer sizes restricted to even numbers
    * Both vectors (position and offset) are very linked
  * Two networks simultanious
    * ...
  * Two networks alternating
    * ...
  * Same network, separate weights
    * Similar to how the weights for the relations were split up

## Combining enitity boxes and Query boxes
* Box (R-)GCN as encoder and box query-answerer as decoder.
  * What loss-function?
* Teach the components alternatingly.
  * What loss-function?
  * Expectation-maximization?
* Teach both components seperately.
  * What loss-function?
  * Seperate decoder for (R-)GCN?

## Loss function
* Distance-based
  * 2 cases
    * 3 distance types
      * Query-centre to overlap		(To overlap-corner if l1-norm is used for the distance metric)
      * Overlap-distance		(From and to overlap-corner if l1-norm is used for the distance metric)
      * Overlap to entity-center	(From overlap-corner if l1-norm is used for the distance metric)
    * 3 distance types
      * Query-centre to query-corner
      * Query-corner to entity-corner
      * Entity-corner to entity-center
  * Freeze entity embeddings that are not part of the desired query result and do not overlap with the query embedding?

## R-GCN update rule
\sigma (\sum_{r \in R} (W_r^{(l)} H^{(l)} A_r D_r) + W_0 H^{(l)})

With **H^{(l)}** being the current hidden layer, **H^{(l+1)}** the next, **W_r^{(l)}** as in equation 2 from [1], **A_r** as the adjacency matrix under **r**, and **W_0** as in equation 2 from [1]. **D_r** is a diagonal, square matrix, with the values **\frac{1}{c_{i,r}}** (as seen in equation 2 form [1]) on the diagonal.

## References
[1] https://arxiv.org/pdf/1703.06103.pdf