# election_pivot

![a0](https://cloud.githubusercontent.com/assets/20483656/20196294/e73d818a-a74e-11e6-830d-e418be148712.png)

Political theory suggests that US Presidential candidates will modify their campaign messages as they shift from targeting primary voters to the wider electorate. This shift is known as the general election pivot. This election has been anything but conventional. I wonder, has this fundamental piece of political theory held true in the world of Donald Trump and Hillary Clinton?

## Data

I gathered the presidential debate transcripts from the 20 primary (11 Republican & 9 Democratic) and 3 general election debates. These transcripts contain relatively unscripted insights into each candidate’s issue priorities and serve as an indicator of a campaign’s overall messaging. I chose four sub-questions I hoped would shed light on my overarching question:

- Debate Dynamics: Has the assertiveness of each candidate changed?
- Sentiment Analysis: Has the sentiment of each candidate changed?
- Summarizer: Have the main talking points of each candidate changed?
- Topic Detection: Have the general topics of each candidate changed?

These analyses were executed using various Natural Language Processing (NLP) and Machine Learning (ML) techniques. Please see the technical appendix for in-depth explanations of each approach. I have also attached a PowerPoint presentation that walks through the topline conclusions of each question.

![a1](https://cloud.githubusercontent.com/assets/20483656/20196386/5c4a3b76-a74f-11e6-9f2c-6f2aa0e606ff.png)

How can you compare the assertiveness of a candidate in a debate with nine opponents to that of a candidate with a single opponent? I created a measure of assertiveness that normalizes for the number of candidates at each debate (see appendix). Although this measure does not make the primary experiences of Clinton and Trump identical, it does allow for general comparisons between each candidate’s debate strategies.

**Clinton was relatively more assertive early in the primary season** and underperformed other candidates since January, accelerating this trend into the general election. **Trump, on the other hand, dominated throughout the primary and general election seasons.** Both candidates, particularly Clinton, experienced a shift in the general election, suggesting some amount of a general election pivot.

Now that we understand how the debate strategies of each candidate changed, what can be understood from the type of language used by each candidate?

![a2](https://cloud.githubusercontent.com/assets/20483656/20196394/689bdede-a74f-11e6-97b2-9cb32a8c2965.png)

The solid line represents a candidate’s subjectivity while the dashed line indicates a candidate’s positivity. Notably, **Trump scores consistently higher levels of subjectivity and lower levels of positivity.** This conclusion would certainly be heralded by Clinton supporters who champion what they believe is a more positive, fact-based campaign. While the two candidate’s subjectivity readings converge by the final debate, their positivity levels diverge in the general election. Intuition would suggest that both candidates have become more negative in the general election; however, this data seems to suggest that **Clinton held her more positive primary campaign message steady through the final presidential debate.** As will be explored later in my comparative analysis, this trajectory is rare, if not unprecedented in recent elections.

These shifts in language and tone will add color to our next question: how have the main talking points of each candidate changed?

## 3) Summarizer

Automatic summary extraction techniques work to string together the most meaningful sentences from a textual corpus to build a summary. I was able to compile four 200 word summaries that capture pivotal moments from each candidate’s time on stage during the primaries and general election. For brevity, I have focused on Trump’s two summaries (emphasis added) and included Clinton’s in the appendix.

### Trump Primaries:

*I say not in a braggadocios way, **I’ve made billions and billions of dollars** dealing with people all over the world, and I want to put whatever that talent is to work for this country so we have great trade deals, we make our country rich again, we make it great again. Many of the great business people that you know and Carl Icon is going to work with me on making great deals for this country… I will know more about the problems of this world by the time I sit, and you look at what’s going in this world right now by people that supposedly know, this world is a mess. **I’ve gained great respect for many and I’m going to even say I mean,** in different forms for the people on the dais, in different forms. And I think after 50 years, and **I have many friends,** I own many properties in Miami, many, many, and I have many people that know and they feel exactly the way I do, make a deal, it would be great, but it’s got to be a great deal for the United States, not a bad deal for the United States.*

### Trump General Election:

*When I watch the deals being made, when I watch what’s happening with some horrible things like **Obamacare,** where your health insurance and health care is going up by numbers that are astronomical, 68 percent, 59 percent, 71 percent, when I look at the **Iran deal** and how bad a deal it is for us, it’s a one-sided transaction where we’re giving back $150 billion to a terrorist state, really, the number one terror state, we’ve made them a strong country from really a very weak country just three years ago. But I don’t want to have, with all the problems this country has and all of the problems that you see going on, hundreds of thousands of people coming in from Syria when we know nothing about them. And I will tell you very strongly, when Bernie Sanders said **she had bad judgment,** she has really bad judgment, because we are letting people into this country that are going to cause problems and crime like you’ve never seen.*

It appears that Trump shifts his message from his personal background and record to a rebuke of Clinton’s policies and judgment. This conclusion matches intuition from listening to the debates.

![a4a](https://cloud.githubusercontent.com/assets/20483656/20196413/776bd932-a74f-11e6-846c-509f3882be57.png)

I used unsupervised ML techniques to explore the core topics in my textual corpus. These models can be instructive at finding both a) how many topics exist in a corpus and b) what those topics are. Rather than subjectivity observing the content of the primaries and comparing it to that of the general election, **this process aims to objectively reveal** the clusters of topics in each corpus.

Trump’s analysis is notable in that he coalesces around a concentrated set of 4 topics across all of the primary debates. **As he transitions to the general election, his message becomes broader** to include topics such as Law & Order and the Supreme Court. The Supreme Court, in particular, is a message that polls favorably for Republicans nationwide and ranks as “very important” issue for 65% of registered voters.[i] Whether or not the inclusion of this topic was intentional, it likely bolstered Trump’s case with traditional Republican voters.

![a4b](https://cloud.githubusercontent.com/assets/20483656/20196428/810f028e-a74f-11e6-9a64-283ffd771c40.png)

Clinton, on the other hand, has a more conventional set of 8 topics in the primaries and 8 in the general election. Whereas Trump had only one of his primary topics drop in the transition to the general election, Clinton had a majority (5 of 8) of her topics change. It would appear that this was a strategic move to pivot from issues important to Democratic primary voters (Gun Control, Education, Financial Regulation) to those that motivate anti-Trump voters (Trump’s Record, Trump’s Temperament, Nuclear Policy).

**These movements by both candidates could have been the result of intentional pivoting or forced pivoting by each candidate’s opponent or the moderator.** In both cases, they reveal a general election race that is changed from the primaries and seems to be more focused on issues that draw distinction between the candidates.

Stepping back and looking at these answers, it’s clear *something* has changed; yet, how can we evaluate if this change is significant?

![a5](https://cloud.githubusercontent.com/assets/20483656/20196444/8bd6fb7c-a74f-11e6-84f8-dfa551dc9f5f.png)

Quantifiably, what is a general election pivot? This qualitative judgment is used without any firm standard for what it means to adapt campaign messaging in the general election. I analyzed an additional 27 debates from the Obama-McCain election (2008) and 20 debates from the Gore-Bush election (2000) to see how 2016 compares.

The magnitude and directionality of the changes in subjectivity and assertiveness from 2016 are similar to those from 2008 and 2000. However, **Clinton seems to be the only candidate** from these three elections **to have increased her message of positivity in the general election.** All told, this analysis confirms that the pivots experienced in 2016 are comparable to those of our two most recent elections without incumbent presidents.

-:-:-:-

For all of this election’s quirks and unprecedented swings, there is still reason to believe that **there has been some form of a general election pivot.** Much of this analysis has confirmed what intuition would predict is true about this election. However, there are **three main conclusions that contradict such assumptions** and are worth noting:

- “When they go low, we go high” – Clinton’s embrace of this quote from Michelle Obama seems to be reflected in her words as well as her strategy through the general election.
- Clinton completely reworked the core topics of her campaign in the general election; yet, she was able to maintain a consistently positive and aspirational message throughout her entire campaign.
- General Election “pivots” are small in magnitude when studied through the lenses of positivity and subjectivity measures, but may be larger in terms of topic modeling.

If you would like to view the full presentation that accompanies this analysis, please [click here.](https://drive.google.com/file/d/0BxUN_Xqt94sSNFUtb2t0V2NjZVk/view "Google Drive")

Thank you for reading.

-:-:-:-

**Appendix:**

In the future, I plan on extending my topic detection analysis to the 2008 and 2000 elections. I would also like to include a few more additional pre-2000 elections to increase the sample size of my comparative analysis.

Technical Notes:

**Question 1:**

I used the following formula to compute assertiveness values for each candidate. This approach normalizes for the number of debate participants, which varies widely debate by debate.

![snip20161031_37](https://cloud.githubusercontent.com/assets/20483656/20196717/d9a2d596-a750-11e6-880f-e35cac5379e0.png)

**Question 2:**

The subjectivity reading was calculated using TextBlob’s Python library. Each word in the English lexicon is given a score for subjectivity, polarity, and intensity. Subjectivity proved the be the most useful measure to differentiate the candidates. It is measured on a 0 (objective) to  1 (subjective) scale. Reference [TextBlob Sentiment: Calculating Polarity and Subjectivity](http://planspace.org/20150607-textblob_sentiment/ "TextBlob Article") for a more in-depth analysis.

The positivity reading was calculated using NLTK’s Naïve Bais Classifier. It was accessed using the NLTK API found here. Whereas TextBlob’s analysis treats each word as independent, this approach is more contextually aware as it trains across a set of documents. The positivity metric is a probability reading (eg. 0.6 indicates a 60% chance that the given statement is positive). There is also a level of “confidence” ascribed to each statement which supports more effective validation of this technique.

**Question 3:**

Alex Perrier did an excellent job explaining the three-step core process behind summary extraction:
- Create an intermediate representation of the text
- Compare and score the sentences
- Build a summary
I used the Summa python library to compute my summaries. This package works well out-of-the-box and requires minimal tuning. It uses TextRank, a graph-based ranking model, to compile the summaries.

**Clinton Primaries**

*I’ve heard a lot about me in this debate, and I’m going to keep talking and thinking about all of you because ultimately, I think the president’s job is to do everything possible, everything that she can do to lift up the people of this country. So if — if people who are in the private sector know what I stand for, it’s what I fought for as a senator, it’s what I will do as president, and they want to be part of once again building our economy so it works for everybody, more power to them, because they are the kind of business leaders who understand that if we don’t get the American economy moving and growing, we’re not going to recognize our country and we’re not going to give our kids the same opportunities that we had. So I see as president having a constant dialogue with Americans here’s what we’re trying to get done, here’s why I need your help, here’s why you may think comprehensive immigration reform with a path to citizenship isn’t something you care about, but I’m telling you it will help fix the labor market, it will bring people out of the shadows.*

**Clinton General Election**

*But it is important for us as a policy, you know, not to say, as Donald has said, we’re going to ban people based on a religion. But I want to on behalf of myself, and I think on behalf of a majority of the American people, say that, you know, our word is good. And on the day when I was in the Situation Room, monitoring the raid that brought Osama bin Laden to justice, he was hosting the “Celebrity Apprentice.” So I’m happy to compare my 30 years of experience, what I’ve done for this country, trying to help in every way I could, especially kids and families get ahead and stay ahead, with your 30 years, and I’ll let the American people make that decision. I think it’s really up to all of us to demonstrate who we are and who our country is, and to stand up and be very clear about what we expect from our next president, how we want to bring our country together, where we don’t want to have the kind of pitting of people one against the other, where instead we celebrate our diversity, we lift people up, and we make our country even greater.*

**Question 4:**

I used Hierarchical Clustering and K-Means to find the optimum number of stable topic clusters for each candidate. I ran into problems particularly with Trump in separating his topics due to his colloquial vocabulary. I was able to tune the number of n-grams for each candidate to overcome this problem. I experimented with dimensionality reduction through PCA and NMF, but ultimately my unreduced feature set provided the most intelligible response.

**Acknowledgments:**

If you’re interested in taking a closer look at the primaries, check out Alex Perrier’s excellent article: “Dissecting the Presidential Debates with an NLP Scalpel”

[i] http://www.people-press.org/2016/07/07/4-top-voting-issues-in-2016-election/
