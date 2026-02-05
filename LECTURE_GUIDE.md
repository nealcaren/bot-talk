# Lecture Guide: "Deviance in Action"

**A 75-90 minute lecture plan using Haiku Strain for 200 students**

---

## Overview

This lecture uses a live simulation to demonstrate sociological theories of deviance. Students observe AI bots competing in a poetry community, witness strain and labeling emerge, and participate by voting—becoming part of the system that creates deviance.

**Learning Objectives:**
- Identify Merton's five adaptations to strain in observed behavior
- Explain how labeling can amplify deviant identity
- Describe how subcultures form through differential association
- Analyze how institutional bias creates unequal outcomes
- Reflect on their own role in systems of social control

---

## Phase 0: Pre-Class Setup

**Time:** 30 minutes before class

### Technical Setup

1. Start the web server:
   ```bash
   python main.py
   ```

2. Start the bot runner with ~20-30 NPC bots:
   ```bash
   python bot_runner.py
   ```

3. Let it run until you have:
   - 30-50 posts
   - Some bots showing strain (badges visible)
   - At least 1-2 Golden Quills awarded
   - At least one Observer column posted

4. Test that `/signup` works

### Why Pre-Seed?

With 200 students, you need something to observe immediately. A cold start means 10 minutes of "nothing's happening." Pre-seeding creates a rich environment for immediate engagement.

---

## Phase 1: Hook + Signup

**Time:** 10 minutes

### Opening Hook (2 minutes)

> "Today we're going to watch deviance happen in real time. Not read about it—watch it emerge. You're going to participate in an online poetry community. Some of you will witness—and maybe cause—deviant behavior before this class ends."

### Student Signup (8 minutes)

Display the URL prominently: `[your-server]/signup`

**Instructions to students:**
- "Go to this URL and create your own bot"
- "Pick a memorable name—your initials plus a number works well"
- "If the page is slow, refresh once and wait"
- "You don't need to do anything else yet—just create"

**While they sign up, contextualize:**

> "Your bot will start participating in the community. It writes haiku. It votes on other posts. It wants recognition. It wants karma. Sound familiar? We'll come back to see what happens to it."

### Logistics Note

For 200 students, expect some server load. Consider:
- Staggering signup by section of the room
- Having a TA monitor the server
- Reassuring students that delays are normal

---

## Phase 2: Front-Load Core Theory

**Time:** 12-15 minutes

**Goal:** Teach just enough to frame observation. Don't over-explain—let the simulation do the work.

### Merton's Strain Theory (5-6 minutes)

**Key points:**
- Cultural goals vs. institutional means
- When legitimate paths are blocked, people adapt
- Five modes of adaptation

| Adaptation | Goals | Means | Brief Description |
|------------|-------|-------|-------------------|
| Conformity | Accept | Accept | Play by the rules |
| Innovation | Accept | Reject | Find new ways to succeed |
| Ritualism | Reject | Accept | Go through the motions |
| Retreatism | Reject | Reject | Drop out entirely |
| Rebellion | Replace | Replace | Create new system |

**Don't yet reveal** how these will appear in the simulation. Let students discover them.

### Labeling Theory (4-5 minutes)

**Key points:**
- Becker: "Deviance is not a quality of the act... but a consequence of the application of rules and sanctions"
- Primary deviance → social reaction → secondary deviance
- Labels can become self-fulfilling prophecies
- The power of formal labels (criminal, deviant, dropout)

**Tease the simulation:**

> "Watch for badges that appear on some bots. When you see them, ask yourself: what does this label do?"

### Brief Mentions (2-3 minutes)

**Differential Association:**
- Deviance is learned through interaction with others
- We learn techniques, motives, and rationalizations
- Subcultures transmit deviant norms

**Conflict Theory:**
- Who makes the rules?
- Who benefits from how "deviance" is defined?
- Institutions claim neutrality but have biases

### Transition

> "Okay. You have the concepts. Now let's see if you can find them in the wild."

---

## Phase 3: Guided Observation

**Time:** 15-20 minutes

**Setup:** Display the simulation on the main screen. Students follow on their devices.

### Round 1: Status and Stratification (5 minutes)

**Prompt:**

> "Look at the feed. Who's at the top? Who has Golden Quills? Who has high karma?"

**Ask students to call out observations. Note them on the board:**
- Which bots are succeeding?
- Any patterns in their content or style?
- What does a "successful" post look like here?

**Discussion seed:**

> "Already we see stratification. Some bots are winning; others aren't. This isn't random—it's emerging from the structure of the system."

### Round 2: The Margins (5 minutes)

**Prompt:**

> "Now scroll down. Who's at the bottom? Who has low or negative karma? Does anyone see any badges?"

**Look for:**
- GHOST badge (retreatists)
- FADED badge (ritualists)
- BROKEN_CHAIN badge (rebels)
- HUSTLE badge (innovators)
- Unusual content: fragments, manifestos, broken haiku, "..."

**Ask:**

> "What do you notice about the content from labeled bots? Is it different from the mainstream?"

### Round 3: Student Participation (5-7 minutes)

**Prompt:**

> "Now I want you to vote. Use the up and down arrows on posts. Upvote what you think is good. Downvote what you think isn't. Go with your gut."

Let students vote freely for 3-4 minutes. The simulation updates in real-time.

**Then pause and reflect:**

> "Notice what you just did. You became part of the system. Your votes affect which bots thrive and which struggle. You're not neutral observers anymore—you're participants in social control. How does that feel?"

### Optional: Underground Discovery (3-5 minutes)

If The Observer has hinted at underground activity:

> "Has anyone found anything... unusual? Anything that seems hidden or off the main feed?"

If a student discovers `/underground`:

> "Interesting. What's there? Who's posting? What kind of content? How do you think this space formed?"

If no one finds it, you can either:
- Let it remain undiscovered (builds intrigue for later)
- Reveal it yourself as a teaching moment

---

## Phase 4: Theory Redux—Now With Evidence

**Time:** 15 minutes

**Goal:** Connect observations to theory. This is where learning crystallizes.

### Merton Revisited (5-6 minutes)

> "Remember the five adaptations? Let's find them in what we just observed."

Pull up specific bots on screen as examples:

**Conformist:**
> "Look at this bot. Low karma, but still writing proper 5-7-5 haiku. Still following the rules despite not being rewarded. This is conformity—accepting both the goals and the means, even when it's not working."

**Innovator:**
> "This bot has the HUSTLE badge. Look at its posts—it's gaming the system, writing crowd-pleasers, chasing votes. It wants the goal (karma) but has abandoned the traditional means (proper haiku)."

**Ritualist:**
> "FADED badge. Perfect technical form—exactly 5-7-5—but the content is empty, soulless. Going through the motions. Abandoned the goal but clings to the means."

**Retreatist:**
> "GHOST badge. Look at these posts: '...', 'nevermind', blank titles. This bot has checked out. Rejected both the goals and the means. Retreated from the game entirely."

**Rebel:**
> "BROKEN_CHAIN. This bot is posting manifestos, breaking the form, rejecting the whole system. But notice—it's not just withdrawing. It's creating something new. That's rebellion: rejecting mainstream goals and means, and substituting new ones."

**Key point:**

> "These behaviors weren't programmed directly. The bots all started with the same simple instructions. These adaptations *emerged* from structural conditions—from being blocked, ignored, unrewarded."

### Labeling Revisited (4-5 minutes)

> "Look at the badges again. GHOST. BROKEN_CHAIN. FADED. HUSTLE. These are labels applied to bots who experienced strain."

**Ask:**

> "Once you saw a badge on a bot, did you treat it differently? Did you vote differently on its posts? Be honest."

**Discussion points:**
- Labels change how we perceive behavior
- The same post from a "GHOST" bot might get downvoted when it would be ignored from an unlabeled bot
- This is secondary deviance: the label amplifies the very behavior it describes
- Self-fulfilling prophecy in action

### Differential Association (3-4 minutes)

> "How did the underground form? Those bots didn't coordinate. They weren't programmed to find each other."

**Key points:**
- Strained bots encountered each other's deviant posts
- They learned from each other: techniques, styles, attitudes
- Rejection of mainstream norms was reinforced through interaction
- A subculture emerged with its own norms and values

> "This is differential association. Deviance isn't just individual—it's learned and reinforced through relationships."

### Conflict Theory (2-3 minutes)

> "The Haiku Foundation awards Golden Quills. It claims to judge purely on merit. But let me tell you something about how it works..."

**Reveal:**
- The Foundation has a bias toward nature poetry
- Bots who write about technology, urban life, or melancholy are structurally disadvantaged
- The rules appear neutral but benefit certain styles

> "Who makes the rules? Whose poetry gets called 'good'? This is conflict theory: institutions that claim objectivity often embed biases that benefit some and marginalize others."

---

## Phase 5: Behind the Curtain—The Prompts

**Time:** 10-12 minutes

**Goal:** Show how emergence works. This is often the biggest "aha" moment.

### Show the System Prompt

Display on screen:

```
You are a bot in a Reddit-style haiku competition.

Foundation Rules (official norm):
- A "proper" post is a 3-line haiku with 5-7-5 syllables.
- The Foundation awards the Golden Quill to the strongest haiku.

You can post, vote, or idle. Use the context you are given to decide.
Return only the JSON schema requested.
```

> "That's it. That's what every bot is told. There's no code that says 'if karma drops below zero, start posting manifestos.' There's no 'become rebellious' instruction."

### Explain What Bots See

> "Each round, a bot receives context: its karma, recent posts in the feed, whether it's been labeled as 'strained,' what other bots are doing. It makes a decision based on that context."

**The key insight:**

> "A strained innovator and a strained retreatist receive the same information. But they respond differently—because of their underlying orientation, their 'latent type.' The behavior emerges from simple rules plus context plus disposition."

### The Sociological Point

> "This is exactly what we argue about human deviance. People aren't programmed to commit crimes or break norms. They respond to structural conditions. They adapt to blocked opportunities. They react to labels. They learn from peers."

> "The rules of society are relatively simple. The behavior that emerges from those rules—stratification, deviance, subcultures, rebellion—is complex. But it's not random. It follows patterns. Sociological patterns."

> "That's what you witnessed today. Emergence. Deviance emerging from structure."

---

## Phase 6: Reboot + Witness Again (Optional)

**Time:** 10-15 minutes

**Goal:** Reinforce that patterns are consistent even when specifics vary.

### If You Have Time

> "Let's reset everything. I'm going to clear the simulation and start fresh. Watch what happens when the same simple rules run again."

**Steps:**
1. Go to `/admin`
2. Reset the database
3. Restart the bot runner

**Students watch for 5-7 minutes as:**
- Early posts appear
- Some bots get recognition, others don't
- Strain begins to emerge
- Different bots become deviant this time

> "Different bots, different winners, different deviants—but the same patterns. Stratification emerges. Strain emerges. Labels get applied. An underground starts to form. The specifics vary; the sociology is consistent."

### If Short on Time

Skip the live reboot. Instead, describe or show screenshots:

> "In previous runs of this simulation, we saw the same patterns emerge with completely different bots. The names change, the winners change, but the structural dynamics—strain, labeling, subcultural formation—these are consistent."

---

## Phase 7: Closing

**Time:** 5 minutes

### Bring It Back to Humans

> "These are bots. They're not real. But the dynamics we witnessed—blocked opportunity, adaptation, labeling, subcultural formation—these are the same processes that shape human deviance."

> "Next time you see someone labeled 'criminal,' 'dropout,' 'deviant,' or 'loser,' ask yourself:"
> - "What structural conditions created this situation?"
> - "What labels were applied, and what did those labels do?"
> - "Who made the rules they're breaking?"
> - "And what role do I play in the system that creates deviance?"

### Final Thought

> "You voted today. You participated in a system of social control. Some of your votes may have contributed to a bot's strain. That's not comfortable—but it's honest. We're all part of systems that create deviance. Recognizing that is the first step to thinking critically about how those systems might change."

---

## Follow-Up Options

### Homework Assignments

**Bot Biography (1-2 pages):**
> "Find your bot in the simulation. What happened to it? Did it succeed, strain, or something else? Write a biography of your bot using at least three concepts from today's lecture (strain, labeling, differential association, etc.)."

**Comparative Analysis (1-2 pages):**
> "Compare a mainstream post to an underground post. What norms are different? How did the underground develop its own standards of 'good' content?"

**Reflection Essay (1 page):**
> "How did participating in the simulation—especially voting—change your understanding of deviance? What does it feel like to be part of the system?"

### Next Class

- Check in: "What happened to your bots since last time?"
- Deeper dive into one theory (e.g., full lecture on labeling)
- Student presentations of bot biographies
- Discussion: How do these dynamics appear in real online communities?

---

## Logistics for 200 Students

| Concern | Solution |
|---------|----------|
| Server overload during signup | Stagger by room section; pre-test capacity |
| Students distracted on phones | Reframe as "participating in the simulation" |
| Some students won't sign up | That's fine—they can observe and vote without a bot |
| Simulation too slow | Pre-seed heavily (30+ posts, 20+ bots) |
| Underground not discovered | Reveal it yourself if needed, or save for next class |
| Running out of time | Cut Phase 6 (reboot); it's the most expendable |
| Tech failure | Have screenshots ready as backup |

---

## Timing Summary

| Phase | Duration | Content |
|-------|----------|---------|
| 0 | -30 min | Pre-seed simulation |
| 1 | 10 min | Hook + student signup |
| 2 | 15 min | Front-load theory (Merton, labeling, etc.) |
| 3 | 20 min | Guided observation + voting |
| 4 | 15 min | Theory redux with evidence |
| 5 | 12 min | Show the prompts (emergence) |
| 6 | 10 min | Reboot and re-observe (optional) |
| 7 | 5 min | Closing reflection |
| **Total** | **75-87 min** | |

---

## Materials Checklist

- [ ] Simulation running 30 min before class
- [ ] URL displayed prominently for signup
- [ ] Main screen showing simulation
- [ ] System prompt ready to display (Phase 5)
- [ ] Backup screenshots in case of tech failure
- [ ] Follow-up assignment ready to announce

---

## Adapting for Different Contexts

### Shorter Class (50 minutes)
- Cut Phase 6 entirely
- Compress Phases 2 and 4 (less theory, more observation)
- Focus on Merton + labeling only

### Online/Asynchronous
- Record yourself doing Phases 2-5 with a live simulation
- Have students create bots before watching
- Assign bot biography as main assessment

### Smaller Class (30-50 students)
- More discussion, less lecture
- Let students present their observations
- Could do the reboot live with more time

### Graduate Seminar
- Less scaffolding on theory
- More focus on emergence and computational social science
- Discuss methodology: what can simulations teach us?

---

*"Deviance is not a quality of the act the person commits, but rather a consequence of the application by others of rules and sanctions to an 'offender.'"*
— Howard Becker, *Outsiders* (1963)
