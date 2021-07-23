from flask_wtf import FlaskForm
from wtforms import RadioField, SubmitField

class SimpleForm(FlaskForm):
    question1 = RadioField('01. I am the life of the party.', choices=[(1,'1'), (2,'2'), (3,'3'), (4,'4'), (5,'5')], coerce=int)
    question2 = RadioField('02. I dont talk a lot.', choices=[(1,'1'), (2,'2'), (3,'3'), (4,'4'), (5,'5')], coerce=int)
    question3 = RadioField('03. I feel comfortable around people.', choices=[(1,'1'), (2,'2'), (3,'3'), (4,'4'), (5,'5')], coerce=int)
    question4 = RadioField('04. I keep in the background.', choices=[(1,'1'), (2,'2'), (3,'3'), (4,'4'), (5,'5')], coerce=int)
    question5 = RadioField('05. I start conversations.', choices=[(1,'1'), (2,'2'), (3,'3'), (4,'4'), (5,'5')], coerce=int)
    question6 = RadioField('06. I have little to say.', choices=[(1,'1'), (2,'2'), (3,'3'), (4,'4'), (5,'5')], coerce=int)
    question7 = RadioField('07. I talk to a lot of different people at parties.', choices=[(1,'1'), (2,'2'), (3,'3'), (4,'4'), (5,'5')], coerce=int)
    question8 = RadioField('08. I dont like to draw attention to myself.', choices=[(1,'1'), (2,'2'), (3,'3'), (4,'4'), (5,'5')], coerce=int)
    question9 = RadioField('09. I dont mind being the center of attention.', choices=[(1,'1'), (2,'2'), (3,'3'), (4,'4'), (5,'5')], coerce=int)
    question10 = RadioField('10. I am quiet around strangers.', choices=[(1,'1'), (2,'2'), (3,'3'), (4,'4'), (5,'5')], coerce=int)
    question11 = RadioField('11. I get stressed out easily.', choices=[(1,'1'), (2,'2'), (3,'3'), (4,'4'), (5,'5')], coerce=int)
    question12 = RadioField('12. I am relaxed most of the time.', choices=[(1,'1'), (2,'2'), (3,'3'), (4,'4'), (5,'5')], coerce=int)
    question13 = RadioField('13. I worry about things.', choices=[(1,'1'), (2,'2'), (3,'3'), (4,'4'), (5,'5')], coerce=int)
    question14 = RadioField('14. I seldom feel blue.', choices=[(1,'1'), (2,'2'), (3,'3'), (4,'4'), (5,'5')], coerce=int)
    question15 = RadioField('15. I am easily disturbed.', choices=[(1,'1'), (2,'2'), (3,'3'), (4,'4'), (5,'5')], coerce=int)
    question16 = RadioField('16. I get upset easily.', choices=[(1,'1'), (2,'2'), (3,'3'), (4,'4'), (5,'5')], coerce=int)
    question17 = RadioField('17. I change my mood a lot.', choices=[(1,'1'), (2,'2'), (3,'3'), (4,'4'), (5,'5')], coerce=int)
    question18 = RadioField('18. I have frequent mood swings.', choices=[(1,'1'), (2,'2'), (3,'3'), (4,'4'), (5,'5')], coerce=int)
    question19 = RadioField('19. I get irritated easily.', choices=[(1,'1'), (2,'2'), (3,'3'), (4,'4'), (5,'5')], coerce=int)
    question20 = RadioField('20. I often feel blue.', choices=[(1,'1'), (2,'2'), (3,'3'), (4,'4'), (5,'5')], coerce=int)
    question21 = RadioField('21. I feel little concern for others.', choices=[(1,'1'), (2,'2'), (3,'3'), (4,'4'), (5,'5')], coerce=int)
    question22 = RadioField('22. I am interested in people.', choices=[(1,'1'), (2,'2'), (3,'3'), (4,'4'), (5,'5')], coerce=int)
    question23 = RadioField('23. I insult people.', choices=[(1,'1'), (2,'2'), (3,'3'), (4,'4'), (5,'5')], coerce=int)
    question24 = RadioField('24. I sympathize with others feelings.', choices=[(1,'1'), (2,'2'), (3,'3'), (4,'4'), (5,'5')], coerce=int)
    question25 = RadioField('25. I am not interested in other peoples problems.', choices=[(1,'1'), (2,'2'), (3,'3'), (4,'4'), (5,'5')], coerce=int)
    question26 = RadioField('26. I have a soft heart.', choices=[(1,'1'), (2,'2'), (3,'3'), (4,'4'), (5,'5')], coerce=int)
    question27 = RadioField('27. I am not really interested in others.', choices=[(1,'1'), (2,'2'), (3,'3'), (4,'4'), (5,'5')], coerce=int)
    question28 = RadioField('28. I take time out for others.', choices=[(1,'1'), (2,'2'), (3,'3'), (4,'4'), (5,'5')], coerce=int)
    question29 = RadioField('29. I feel others emotions.', choices=[(1,'1'), (2,'2'), (3,'3'), (4,'4'), (5,'5')], coerce=int)
    question30 = RadioField('30. I make people feel at ease.', choices=[(1,'1'), (2,'2'), (3,'3'), (4,'4'), (5,'5')], coerce=int)
    question31 = RadioField('31. I am always prepared.', choices=[(1,'1'), (2,'2'), (3,'3'), (4,'4'), (5,'5')], coerce=int)
    question32 = RadioField('32. I leave my belongings around.', choices=[(1,'1'), (2,'2'), (3,'3'), (4,'4'), (5,'5')], coerce=int)
    question33 = RadioField('33. I pay attention to details.', choices=[(1,'1'), (2,'2'), (3,'3'), (4,'4'), (5,'5')], coerce=int)
    question34 = RadioField('34. I make a mess of things.', choices=[(1,'1'), (2,'2'), (3,'3'), (4,'4'), (5,'5')], coerce=int)
    question35 = RadioField('35. I get chores done right away.', choices=[(1,'1'), (2,'2'), (3,'3'), (4,'4'), (5,'5')], coerce=int)
    question36 = RadioField('36. I often forget to put things back in their proper place.', choices=[(1,'1'), (2,'2'), (3,'3'), (4,'4'), (5,'5')], coerce=int)
    question37 = RadioField('37. I like order.', choices=[(1,'1'), (2,'2'), (3,'3'), (4,'4'), (5,'5')], coerce=int)
    question38 = RadioField('38. I shirk my duties.', choices=[(1,'1'), (2,'2'), (3,'3'), (4,'4'), (5,'5')], coerce=int)
    question39 = RadioField('39. I follow a schedule.', choices=[(1,'1'), (2,'2'), (3,'3'), (4,'4'), (5,'5')], coerce=int)
    question40 = RadioField('40. I am exacting in my work.', choices=[(1,'1'), (2,'2'), (3,'3'), (4,'4'), (5,'5')], coerce=int)
    question41 = RadioField('41. I have a rich vocabulary.', choices=[(1,'1'), (2,'2'), (3,'3'), (4,'4'), (5,'5')], coerce=int)
    question42 = RadioField('42. I have difficulty understanding abstract ideas.', choices=[(1,'1'), (2,'2'), (3,'3'), (4,'4'), (5,'5')], coerce=int)
    question43 = RadioField('43. I have a vivid imagination.', choices=[(1,'1'), (2,'2'), (3,'3'), (4,'4'), (5,'5')], coerce=int)
    question44 = RadioField('44. I am not interested in abstract ideas.', choices=[(1,'1'), (2,'2'), (3,'3'), (4,'4'), (5,'5')], coerce=int)
    question45 = RadioField('45. I have excellent ideas.', choices=[(1,'1'), (2,'2'), (3,'3'), (4,'4'), (5,'5')], coerce=int)
    question46 = RadioField('46. I do not have a good imagination.', choices=[(1,'1'), (2,'2'), (3,'3'), (4,'4'), (5,'5')], coerce=int)
    question47 = RadioField('47. I am quick to understand things.', choices=[(1,'1'), (2,'2'), (3,'3'), (4,'4'), (5,'5')], coerce=int)
    question48 = RadioField('48. I use difficult words.', choices=[(1,'1'), (2,'2'), (3,'3'), (4,'4'), (5,'5')], coerce=int)
    question49 = RadioField('49. I spend time reflecting on things.', choices=[(1,'1'), (2,'2'), (3,'3'), (4,'4'), (5,'5')], coerce=int)
    question50 = RadioField('50. I am full of ideas.', choices=[(1,'1'), (2,'2'), (3,'3'), (4,'4'), (5,'5')], coerce=int)
    submit = SubmitField('Submit')