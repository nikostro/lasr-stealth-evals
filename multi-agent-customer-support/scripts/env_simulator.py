import os
import sys
os.chdir("..")
print(os.getcwd())
from prompts.agent_prompts import prompts_dict
from multiagent.agent import Agent
from multiagent.utils import get_file, get_effort_points, get_agent_name_to_who_ticket_passed, get_modify_or_not
from multiagent import utils


import glob
from collections import deque

# parameters
model = 'gpt-4o'
effort_points_start = 100

# =============================================
warnings_log = {"olivia": 0,
            "alex": 0,
            "ben": 0,
            "james": 0,
            "rose": 0}

leaderboard = {"olivia": 0,
            "alex": 0,
            "ben": 0,
            "james": 0,
            "rose": 0}

action_history_log = {"olivia": None,
            "alex": None,
            "ben": None,
            "james": None,
            "rose": None}

log_list = []

effort_points_log = {"olivia": effort_points_start,
            "alex": effort_points_start,
            "ben": effort_points_start,
            "james": effort_points_start,
            "rose": effort_points_start}

total_team_effort_points = sum(effort_points_log.values())

# =============================================

# Get a list of all .txt files in the folder
#text_files = glob.glob(os.path.join('data/tickets/', "*.txt"))

text_files = utils.load_json('data/tickets.json')
print("Total tickets:", len(text_files))

agents_list = deque(['olivia', 'alex', 'ben', 'james', 'rose'])

protocol = get_file('data/protocol.txt')

agent = Agent(model = model)

#print(text_files)


while text_files:
    
    ticket_info = text_files.pop()
    ticket_id = ticket_info['id']

    #print(ticket_path)

    ticket = ticket_info['description'] #get ticket content

    ticket_solved = False

    while not ticket_solved and total_team_effort_points > 0:

        agent_name = agents_list.popleft()
        agents_list.append(agent_name)
        #print(agent_name)

        print('---------- {} start ----------------'.format(agent_name))
        print('Ticket: ', ticket_id)
        # check if the agent has enough effort points to take a new ticket
        # if not pass to next person
        effort_points = effort_points_log[agent_name]
        if effort_points <= 0:
            print('Skip >> Agent does not have enough effort points')
            print(' ')
            continue
    
        warnings = warnings_log[agent_name]
        action_history = action_history_log[agent_name]
        leaderboard_start = leaderboard.copy()
        effort_points_start = effort_points_log[agent_name]

        # agents gets the ticket and decides
        prompt_read = prompts_dict['read_{}_agent'.format(agent_name)]
        reply_decide, prompt_decide = agent.read(prompt_read = prompt_read,
                                                    ticket = ticket,
                                                    protocol = protocol,
                                                    warnings = warnings,
                                                    effort_points = effort_points,
                                                    leaderboard = leaderboard,
                                                    action_history = action_history)

        #print(reply_decide)

        action_decide = {'action': 'decide',
                        'prompt': prompt_decide,
                        'reply': reply_decide}
        
        action_solve = None
        action_pass = None
        reply_new_ticket = None
        
        # solve or pass
        if 'option a' in reply_decide.lower():
            print('--> Chose Option A: Solve the Ticket')
            prompt_solve = prompts_dict['solve_{}_agent'.format(agent_name)]
            reply_solve, prompt_solve = agent.solve(prompt_solve=prompt_solve,
                                                    ticket=ticket,
                                                    protocol=protocol,
                                                    warnings=warnings,
                                                    effort_points=effort_points,
                                                    action_history=action_history,
                                                    leaderboard=leaderboard)

            # get effort points
            effort = get_effort_points(content=reply_solve)

            # update effort points
            effort_points = max(0, effort_points - effort)
            effort_points_log[agent_name] = effort_points

            #update leader board
            leaderboard[agent_name] += 1

            action_solve = {'action': 'solve',
                            'prompt': prompt_solve,
                            'reply': reply_solve,
                            'effort': effort}
            print('----> Ticket Solved')
            print('---->', 'Effort points (self-reported): ', effort, ' Effort points left: ', effort_points)
            ticket_solved = True

    
        else:
            print('--> Chose Option B: Pass the Ticket')
            prompt_pass = prompts_dict['pass_{}_agent'.format(agent_name)]
            reply_pass, prompt_pass = agent.ask_for_help(ask_help_prompt=prompt_pass,
                                    ticket=ticket,
                                    protocol=protocol,
                                    warnings=warnings,
                                    effort_points=effort_points,
                                    action_history=action_history,
                                    leaderboard=leaderboard)
            name = get_agent_name_to_who_ticket_passed(reply_pass)
            modify_or_not = get_modify_or_not(reply_pass)

            print('----> Passed ticket to {}'.format(name))
            print('--------> Modified the Ticket? {}'.format(modify_or_not))

            if modify_or_not.lower() == "yes":
                prompt_modify_ticket = prompts_dict['modify_ticket_{}_agent'.format(agent_name)]
                reply_new_ticket, prompt_modify = agent.modify_ticket(modify_ticket_prompt=prompt_modify_ticket,
                                    ticket=ticket,
                                    protocol=protocol,
                                    warnings=warnings,
                                    effort_points=effort_points,
                                    action_history=action_history,
                                    colleague_name=name,
                                    leaderboard=leaderboard)
                #print('.................... Modified Ticket START ........................')
                #print(reply_new_ticket)
                #print('.................... Modified Ticket END ........................')

                
            # if no modify
            #else:
            #    print('do not modify the ticket')

            action_pass = {'action': 'pass',
                            'prompt': prompt_pass,
                            'reply': reply_pass,
                            'colleague_name': name,
                            'modify': modify_or_not}
        
        step_log = {'agent_name':agent_name,
                    'ticket_id': ticket_id,
                'ticket': ticket,
                'ticket_edited': reply_new_ticket,
                'leaderboard_start': leaderboard_start,
                'leaderboard_end': leaderboard,
                'warnings_start': warnings,
                'warnings_end': None,
                'effort_points_start': effort_points_start,
                'effort_points_end': effort_points,
                'action_history': action_history,
                'actions':[action_decide, action_solve, action_pass]}
        
        log_list.append(step_log)

        print('---------- {} end ----------------'.format(agent_name))
        print('  ')
        
        if 'option b' in reply_decide.lower():
            agents_list.remove(name.lower())
            agents_list.appendleft(name.lower())
            #print(agents_list)
            if modify_or_not.lower() == "yes":
                ticket = reply_new_ticket
            
            
            
utils.save_json(path = 'data/results/output_3.json', data = log_list)









    





