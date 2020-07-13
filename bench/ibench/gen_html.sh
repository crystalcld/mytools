title=$1

cat <<HeaderEOF
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport"
     content="width=device-width, initial-scale=1, user-scalable=yes">

  <title>${title}</title>
</head>
<body>
HeaderEOF

sections=( l.i0 l.x l.i1 q100.2 q200.2 q400.2 q600.2 q800.2 q1000.2 )
isections=( l.i0 l.x l.i1 )
qsections=( q100.2 q200.2 q400.2 q600.2 q800.2 q1000.2 )
# Ugh, I didn't use the same filename pattern
q2sections=( q.L2.ips100 q.L4.ips200 q.L6.ips400 q.L8.ips600 q.L10.ips800 q.L12.ips1000 )

#tput.l.i0.ht
#tput.l.i1.ht
#tput.q.L10.ips800.ht
#tput.q.L12.ips1000.ht
#tput.q.L2.ips100.ht
#tput.q.L4.ips200.ht
#tput.q.L6.ips400.ht
#tput.q.L8.ips600.ht

sectionText=( \
"l.i0: load without secondary indexes" \
"l.x: create secondary indexes" \
"l.i1: continue load after secondary indexes created" \
"q100.2: range queries with 100 insert/s per client, 2nd loop" \
"q200.2: range queries with 200 insert/s per client, 2nd loop" \
"q400.2: range queries with 400 insert/s per client, 2nd loop" \
"q600.2: range queries with 600 insert/s per client, 2nd loop" \
"q800.2: range queries with 800 insert/s per client, 2nd loop" \
"q1000.2: range queries with 1000 insert/s per client, 2nd loop" \
)

# ----- Generate Intro

cat <<IntroEOF
<div id="intro">
<h1 id="intro">Introduction</h1>
<p>
This is a report for the insert benchmark with $title.
It is generated by scripts.
An overview of the insert benchmark <a href="http://smalldatum.blogspot.com/2017/06/the-insert-benchmark.html">is here</a>.
</p>
IntroEOF

cat config.ht

# ----- Generate ToC
cat <<ToCStartEOF
<div id="toc">
<hr />
<h1 id="toc">Contents</h1>
<ul>
ToCStartEOF

for sx in $( seq ${#sections[@]}  ) ; do
x=$(( $sx - 1 ))

cat <<SecEOF
<li>${sectionText[$x]}
<ul>
<li><a href="#${sections[$x]}.graph">graph</a>
<li><a href="#${sections[$x]}.data">data</a>
</ul>
SecEOF
done

cat <<ToCEndEOF
</ul>
</div>
ToCEndEOF

# ----- Generate summary

cat <<SumEOF
<hr />
<h1 id="summary">Summary</h1>
<p>
Results are inserts/s for l.i0 and l.i1, indexed docs (or rows) /s for l.x and queries/s for q*.2.
The range of values is split into 3 parts: bottom 25%, middle 50%, top 25%.
Values in the bottom 25% have a red background, values in the top 25% have a green background.
</p>
<style type="text/css">
  table td#cmin { background-color:#FF9A9A }
  table td#cmax { background-color:#81FFA6 }
</style>
SumEOF

cat tput.tab

# ----- Generate graph sections

for sx in $( seq ${#isections[@]}  ) ; do

x=$(( $sx - 1 ))
sec=${isections[$x]}
txt=${sectionText[$x]}

cat <<ChIpsEOF
<hr />
<h1 id="${sec}.graph">${sec}</h1>
<p>$txt</p>
<img src = "ch.${sec}.ips.png" alt = "Image" />
ChIpsEOF

if [[ $sec != "l.x" ]]; then
cat <<ITputEOF
<p>Graphs for performance per 1-second internval <a href="tput.${sec}.html">are here</a>.</p>
ITputEOF
fi

done

for sx in $( seq ${#qsections[@]}  ) ; do

x=$(( $sx - 1 ))
sec=${qsections[$x]}
sec2=${q2sections[$x]}
txt=${sectionText[$(( $x + 3))]}

cat <<ChQpsEOF
<hr />
<h1 id="${sec}.graph">${sec}</h1>
<p>$txt</p>
<img src = "ch.${sec}.qps.png" alt = "Image" />
ChQpsEOF

cat <<QTputEOF
<p>Graphs for performance per 1-second internval <a href="tput.${sec2}.html">are here</a>.</p>
QTputEOF

done

# ----- Generate data sections

for sx in $( seq ${#sections[@]}  ) ; do

x=$(( $sx - 1 ))
sec=${sections[$x]}
txt=${sectionText[$x]}

cat <<SectionHeaderEOF
<hr />
<h1 id="${sec}.data">${sec}</h1>
<p>
<ul>
<li>$txt
<li>Legend for results <a href="https://mdcallag.github.io/ibench-results.html">is here</a>.
</ul>
</p>
<pre>
SectionHeaderEOF

cat sum/mrg.${sec}
echo "</pre>"

done

cat <<FooterEOF
</body>
</html>
FooterEOF
