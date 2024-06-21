"""This script runs all the lemlab scenarios specified in the scenario folder.
The scenarios already need to be converted from the hamlet formet"""

from telegram import Telegram
from lemlab import ScenarioExecutor
import os
import time
from tqdm import tqdm
import logging
import traceback
import datetime
import sys

# Configure the logging module
logging.basicConfig(
    filename='app.log',
    level=logging.ERROR,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

MACHINES = ['RM1', 'RM2', 'STROM1', 'STROM2', 'T14G3']
MACHINE = MACHINES[-1]  # Machine that is currently worked on


def main(lemlab_scenario: str = './scenarios', lemlab_results: str = './simulation_results'):
    # Get a list of all scenarios to simulate
    scenarios = next(os.walk(lemlab_scenario))[1]
    num_scenarios = len(scenarios)



    # Prepare the progressbar
    pbar = tqdm(total=len(scenarios), unit='scenario')

    # Create a telegram bot
    bot = Telegram()

    # Send a message with time and date
    bot.send_message(f'*STARTING SIMULATION*                \n'
                     f'Device: {MACHINE}                    \n'
                     f'Time: {time.strftime("%H:%M:%S")}    \n'
                     f'Date: {time.strftime("%d.%m.%Y")}    \n'
                     f'Simulations: {num_scenarios}')

    # Prepare the loop variables
    old_date = None                     # Date of the last simulation
    counter = 0                         # Number of simulations finished
    daily_counter = 0                   # Number of simulations finished on the current day
    duration = [time.perf_counter()]    # duration of each simulation with the first element being the start time

    # Go through each scenario
    for scenario in scenarios:
        # Save current date
        date = datetime.datetime.now().strftime("%d.%m.%Y")

        # Send summary message if it is a new day
        if (date != old_date) and old_date:
            avg_runtime = time.strftime("%H:%M:%S", time.gmtime((duration[-1] - duration[0]) / (len(duration) - 1)))
            string = f'*DAILY SUMMARY*                                      \n' \
                     f'Device: {MACHINE}                                    \n' \
                     f'Date: {old_date}                                     \n' \
                     f'Simulations today: {daily_counter}                   \n' \
                     f'Avg. runtime: {avg_runtime}                          \n' \
                     f'Simulations run (total): {counter}/{num_scenarios}   \n'
            bot.send_message(string)

            # Reset daily counter
            daily_counter = 0

        # Excluded as it would spam the telegram channel
        # # Send a message with scenario and current time (H:M:S)
        # string = f'{MACHINE}: Starting {scenario} at {time.strftime("%H:%M:%S")}'
        # bot.send_message(string)

        # Set the progressbar description
        string = f"Running {scenario}"
        pbar.set_description(string)

        try:
            sys.stdout = open(os.devnull, 'w')  # deactivate printing
            # Create a scenario executor
            simulation = ScenarioExecutor(path_scenario=os.path.join(lemlab_scenario, scenario),
                                          path_results=os.path.join(lemlab_results, scenario))
            # Run the scenario
            simulation.run()
            sys.stdout = sys.__stdout__  # reactivate printing
        except Exception as e:
            # Send a message with scenario and current time (H:M:S)
            string = f'{MACHINE}: ERROR in {scenario} at {time.strftime("%H:%M:%S | %d.%m.%Y")} \n' \
                     f'Skipping scenario'
            bot.send_message(string)
            # Log error
            logging.error(traceback.format_exc())
            # Skip scenario
            print(f'\n {string}')
            # raise e
            continue

        # Set time counter
        duration.append(time.perf_counter())

        # Increase counters
        counter += 1
        daily_counter += 1

        # Excluded as it would spam the telegram channel
        # # Send a message with scenario and elapsed time (H:M:S)
        # string = f'{MACHINE}: Finished {scenario} (scenario {counter} of {num_scenarios}) ' \
        #          f'in {time.strftime("%H:%M:%S", time.gmtime(duration[counter] - duration[counter - 1]))}'
        # bot.send_message(string)

        # Update the progressbar
        pbar.update()

        # Save date as old date
        old_date = date

    # # Close the progressbar
    # pbar.close()

    # Send a message when all scenarios are finished
    avg_runtime = time.strftime("%H:%M:%S", time.gmtime((duration[-1] - duration[0]) / (len(duration) - 1)))
    total_runtime = time.strftime("%H:%M:%S", time.gmtime(duration[-1] - duration[0]))
    string = f'*FINAL SUMMARY*                      \n' \
             f'Device: {MACHINE}                    \n' \
             f'Finished: {old_date}                     \n' \
             f'Simulations run: {counter}/{num_scenarios}   \n' \
             f'Avg. runtime: {avg_runtime}       \n' \
             f'Total runtime: {total_runtime}       \n'
    bot.send_message(string)


if __name__ == "__main__":
    main()
