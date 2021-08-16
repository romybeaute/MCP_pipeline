# Instructions
instructions = mne.events_from_annotations(raw)
instructions = instructions[0]
items = instructions[:,2]
values = np.unique(instructions[:,2])
print(values)
for i in range(len(values)):
    count_event_id = np.where(instructions[:,2]==values[i])
    print(‘Id’,values[i], ‘:’,len(count_event_id[0]))
#Creating object
if events_used == [‘right’]:
    event_id = {1: ‘open/right’, 2: ‘close/right’}
    if file == 0: #REC1
        event_key = {197: 1, 133: 2} #onset: {205: 1, 141: 2}
    elif file ==14: #REC2
        event_key = {5 : 1} #onset : {13 : 1}
elif events_used == [‘left’]:
    event_id = {3: ‘open/left’, 4: ‘close/left’}
    if file == 8: #REC1
        event_key = {213: 3, 149: 4} #change 149 try {213: 3, 157: 4}
    elif file ==3: #REC2
        event_key = {21: 4} #onset {29: 3}
# Creation of instructions object
event_dict = {value: key for key, value in event_id.items()}
event = list()
events = list()
events_info = list()
a = 0
instr = 0
for instr_id, (onset, _, code) in enumerate(instructions):
    # Generate 5 epochs
    if code in event_key:
        for repeat in range(n_epo_segments): #n_epo_segments = 5
            event = [onset + repeat * n_samples, 0, 1+int(a)]
            events.append(event)
            events_info.append(instr)
        instr+=1
        a = not a
events = np.array(events, int)