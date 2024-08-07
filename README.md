# stark-anatomy

STARK tutorial with supporting code in python

Outline:
 - introduction
 - overview of STARKs
 - basic tools -- algebra and polynomials
 - FRI low degree test
 - STARK information theoretical protocol
 - speeding things up with NTT and preprocessing

Visit the Github Pages website here: https://aszepieniec.github.io/stark-anatomy/

## Follow-up
- Implementation of a ZK-STARK VM for the esoteric and very simple programming language Brainfuck: [https://github.com/aszepieniec/stark-brainfuck](https://github.com/aszepieniec/stark-brainfuck)
- A feature-complete, efficient, and useful ZK-STARK VM utilizing techniques from this tutorial and from the stark-brainfuck tutorial: [https://github.com/TritonVM/triton-vm](https://github.com/TritonVM/triton-vm)

## Running locally (the website, not the tutorial)

 1. Install ruby
 2. Install bundler
 3. Change directory to `docs/` and install Jekyll: `$> sudo bundle install`
 4. Run Jekyll: `$> bundle exec jekyll serve`
 5. Surf to [http://127.0.0.1:4000/](http://127.0.0.1:4000/)

## LaTeX and Github Pages

Github-Pages uses Kramdown as the markdown processor. Kramdown does not support LaTeX. Instead, there is a javascript header that loads MathJax, parses the page, and replaces LaTeX maths instructions with their proper formulae. Here is how to do it:

1. Open `_includes/head-custom.html` and paste the following code:
```javascript
<script type="text/x-mathjax-config">
MathJax.Hub.Config({
    TeX: {
      equationNumbers: {
        autoNumber: "AMS"
      }
    },
    tex2jax: {
    inlineMath: [ ['$', '$'], ['\\(', '\\)'] ],
    displayMath: [ ['$$', '$$'], ['\\[', '\\]'] ],
    processEscapes: true,
  }
});
MathJax.Hub.Register.MessageHook("Math Processing Error",function (message) {
	  alert("Math Processing Error: "+message[1]);
	});
MathJax.Hub.Register.MessageHook("TeX Jax - parse error",function (message) {
	  alert("Math Processing Error: "+message[1]);
	});
</script>
<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>
```

Jekyll, the site engine used by Github Pages, will load this header automatically. There is no need to change the `_config.yml` file.

Note that Kramdown interprets every underscore (`_`) that is followed by a non-whitespace character, as starting an emphasised piece of text. This interpretation interfereces with subscript in LaTeX formulae, which also uses underscores. The workaround is to re-write the LaTeX formulas by introducing a space after every underscore. Also, consider replacing:
 - `\{` by `\lbrace` and `\}` by `\rbrace`,
 - `|` by `\vert`.

