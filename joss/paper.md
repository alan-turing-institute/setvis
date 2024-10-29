---
title: 'SetVis: Visualizing Large Numbers of Sets and Intersections'
tags:
- Python
- set visualization
- sets
date: "7th May 2024"
output: pdf_document
authors:
- name: R.A. Ruddle
  orcid: 0000-0001-8662-8103
  affiliation: 1, 2
- name: L. Hama
  orcid: 0000-0003-1912-4890
  affiliation: 1
- name: P Wochner
  orcid: 0000-0003-4066-8614
  affiliation: 2
- name: O.T. Strickson
  orcid: 0000-0002-8177-5582
  affiliation: 2
bibliography: paper.bib
affiliations:
- name: University of Leeds, Leeds, United Kingdom
  index: 1
- name: Alan Turing Institute, London, United Kingdom
  index: 2
---

# Summary

Set-type data occurs in many domains such as life sciences [@lamy2019rainbio], health [@landolfi2022screening] and the retail industry [@adnan2018set], as well as in generic applications such as analysing structures of missing data [@ruddle2022using] and association rule mining [@wang2020visual]. SetVis is new matrix-based set visualization software, implemented as a Python package which is available from [PyPi](https://pypi.org/project/setvis). The documentation is available from [setvis.readthedocs.io](https://setvis.readthedocs.io) which contains various hands-on Jupyter notebooks guiding users to use the package. SetVis uses a memory-efficient design, operates with datasets held in RAM or a database such as PostgreSQL, and allows user interaction graphically or programmatically. A technical evaluation shows that SetVis uses orders of magnitude less memory than the UpSet [@lex2014upset] Python package [@nothman2022].

# Statement of need
Although a wide variety of set visualization software has been developed [@jia2021venn;@alsallakh2016powerset], most of such software generates Venn or Euler diagrams so is only suitable for data that contain fewer than ten sets [@jia2021venn]. Other software visualizes 50+ sets but either has little support for set intersection tasks [@alper2011design;@dork2012pivotpaths;@kim2007visualizing;@freiler2008interactive] or only visualizes pairwise intersections [@molbiotools2022;@yalcin2015aggreset].

The best-known software for analysing rich patterns in set data are the R and Python UpSet plot packages [@conway2017upsetr;@nothman2022], but the memory requirement of both packages increases linearly with the number of cells (i.e., rows $\times$ columns) in a dataset, which makes the packages unusable with big data. The ACE software [@ruddle2022ace] uses more memory-efficient data structures, but first requires the whole of a dataset to have been loaded into RAM (again, clearly an issue for big data), and is a stand-alone Java application that cannot be integrated with Jupyter Notebooks or similar workflows. The SetVis python package addresses the above collective weaknesses because it: (a) operates with datasets that may be either held in RAM or out of core (in a PostgreSQL database), (b) stores sets and intersections in memory-efficient data structures (like ACE), (c) can be used within Jupyter Notebooks (or similar) to aid the replicability of analysis workflows, and (d) allows users to interact graphically in a notebook as well as programmatically.

# Design
![An example APC combination heatmap shows the fields (X axis), each combination of missing values (Y axis) and the number of records that are in each combination (colour) of the APC (Admitted Patient Care) dataset included in the package. The top, 4th from top and bottom six combinations are a monotone pattern. However, the other seven combinations show that there is another pattern that has gaps in the DIAG fields. \label{fig:heatmap}](../notebooks/images/combination_heatmap.JPG)

SetVis provides the same six built-in visualizations as ACE [@ruddle2022ace]. The main two show visualizations of sets (in a bar chart) and set intersections (in a heatmap). The other four visualizations make SetVis scalable to data that contains large numbers of sets and/or intersections, by showing histograms of set cardinality, intersection degree and intersection cardinality, and an intersection degree bar chart. All of the visualizations are interactive [implemented with Bokeh, @bokeh2018], but users may also interact programmatically and freely interleave the two forms of interaction. Examples and tutorials are provided with the installation. A screenshot of SetVis version `v0.1rc5` of the heatmap visualizations within Jupyter notebooks is shown in Figure \ref{fig:heatmap}.

Jupyter notebooks have been widely adopted in the Python data science ecosystem for exploratory data analysis. It is considered good practice for computational notebooks to obey principles of (i) top-to-bottom re-executability and (ii) repeatability, including by others [@quaranta2022notebooks]. The SetVis design allows these principles to be respected.

SetVis is underpinned by memory-efficient data structures. Set membership information for each of set membership information for each of $K$ sets can be represented with a mapping from an element (represented by its index) to a tuple of $K$ booleans based on indicator functions for each of these sets:

\begin{equation} \label{eq:members}
members: ElementIndex \to \{True, False\}^K.
\end{equation}

One component of the resulting tuple indicates membership of a particular set. Storing this mapping explicitly [e.g., as in UpSet with a dataframe, @conway2017upsetr;@nothman2022] requires $O(KN)$ storage, where $K$ and $N$ are the number of sets and the number of elements. When $K$ is large, as is the case for many real-world datasets, this can be inefficient.
The number of unique set intersections, $R$, is often much smaller than the number of records, $R \ll N$, and can be at most $N$ (if each element is member of a unique combination of sets). SetVis makes use of this idea, and considers
\begin{equation}
members = intersectionMembers \circ intersectionId
\end{equation}
where
\begin{equation}
intersectionId: ElementIndex \to IntersectionIndex
\end{equation}
maps an element index to an index referring to the particular combination of sets to which that element belongs; and
\begin{equation}
intersectionMembers: IntersectionIndex \to \{True, False\}^K
\end{equation}
is a bijection between an intersection index and the explicit representation of this combination.

In SetVis, these mappings are stored as a pair of Pandas dataframes (in an instance of the \textit{Membership} class), $intersectionId$ of size $O(N)$ and $intersectionMembers$ of size $O(RK)$, for a combined total of $O(N + RK)$ storage.

# Technical evaluation
Using a 44GB Ubuntu virtual machine, we compared SetVis (v0.1.0) with UpSetPlot (v0.8.0) based on two criteria: memory use and compute time. The greatest difference was in memory usage. The UpSetR package [@conway2017upsetr] was not tested, but uses a similar data structure to UpSetPlot[@nothman2022].

Tests were run with set-type data that contained two columns, 500,000 rows and 100 – 10,000 set intersections. UpSetPlot crashed when the 500,000 row dataset contained more than 500 intersections. By contrast, SetVis only used 113 MB RAM for that dataset (see Figure \ref{fig:both}A).

![(A) Memory used by UpSetPlot and SetVis for set-type data with 500,000 rows, two columns and a range of set intersections. There were always 10\% more sets than intersections. (B) Memory used by UpSetPlot and SetVis for visualizing patterns of missing data. The number of cells equals the number of rows $\times$ columns in a dataset. \label{fig:both}](comb.png)

The difference was even more pronounced when the packages were used to analyze missing data (10,000 – 500,000 rows; 10 – 700 columns; each row missing one value). UpSetPlot's memory scaled linearly with the number of cells (i.e., rows $\times$ columns) in a dataset, whereas SetVis's memory only increased gradually (see Figure \ref{fig:both}B). There was a similarly large difference when each row contained 1 – 50 missing values (100 – 10,000 set intersections in each dataset), because for missing data UpSetPlot keeps a copy of the input Pandas dataframe, as well as having a memory-hungry design.


# Acknowledgements

This research was supported by the Alan Turing Institute and the Engineering and Physical Sciences Research Council (EP/N013980/1; EP/R511717/1).

# References
