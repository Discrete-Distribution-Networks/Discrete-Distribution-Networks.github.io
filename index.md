
<head>
    <meta charset="UTF-8">
    <title>DDN: Discrete Distribution Networks - Novel Generative Model</title>
    <meta name="description" content="DDN: Discrete Distribution Networks">
    <meta name="keywords" content="DDN, generative model">
    <link rel="shortcut icon" href="2d_density.1_two-sprials_gen.png">
</head>

<div align="center">

<!-- # Discrete Distribution Networks -->

<p style="font-size: 2em; font-weight: bold; margin-top: 20px; margin-bottom: 7px; line-height: 1;">Discrete Distribution Networks</p>

**A Novel Generative Model with Simple Principles and Unique Properties**

<br>


**[Lei Yang](https://github.com/DIYer22)**

---

### | [Paper]() <sub>(Coming soon)</sub> | [Code]()<sub>(Coming soon)</sub>  |

<!-- 
全新的生成模型, 有着简单的原理和独特的性质
- Code 分为
    - sddn 库
    - toy
    - pretrain
 -->

</div>



## 1. Abstract
<p style="text-align: justify"><em>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
We introduce a novel generative model, the Discrete Distribution Networks (DDN), that approximates data distribution using hierarchical discrete distributions. We posit that since the features within a network inherently contain distributional information, liberating the network from a single output to concurrently generate multiple samples proves to be highly effective. Therefore, DDN fits the target distribution, including continuous ones, by generating multiple discrete sample points. To capture finer details of the target data, DDN selects the output that is closest to the Ground Truth (GT) from the coarse results generated in the first layer. This selected output is then fed back into the network as a condition for the second layer, thereby generating new outputs more similar to the GT. As the number of DDN layers increases, the representational space of the outputs expands exponentially, and the generated samples become increasingly similar to the GT. This hierarchical output pattern of discrete distributions endows DDN with two intriguing properties: highly compressed representation and more general zero-shot conditional generation. We demonstrate the efficacy of DDN and these intriguing properties through experiments on CIFAR-10 and FFHQ.
</em></p>



<style>
html, body {
width: auto !important;
max-width: 100% !important;
padding: 0px !important;
margin: 0px !important;
}
._theme-github {
background-color: rgb(255, 255, 255);
}
.markdown-body {
min-width: 512px;
max-width: 888px;
background-color: rgb(255, 255, 255);
overflow: auto;
border-width: 1px;
border-style: solid;
border-color: rgb(221, 221, 221);
border-image: initial;
padding: 45px;
margin: 20px auto;
}
</style>
