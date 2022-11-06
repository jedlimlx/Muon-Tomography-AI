// ------------------------------------ //
// mu.cc
// Geant4 example for muography
// Developer: Yuri Yoshihara
// 2018 Sep 29
// ------------------------------------ //

#include "muDetectorConstruction.hh"
#include "muPhysicsList.hh"
#include "muPrimaryGeneratorAction.hh"
#include "muRunAction.hh"
#include "muEventAction.hh"
#include "muParticleGun.hh"
#include "muParticleGunMessenger.hh"
#include "muAnalyzer.hh"

#include "muRunManager.hh"

#include "G4UImanager.hh"
#include "G4VisExecutive.hh"
#include "G4UIExecutive.hh"
#include "Randomize.hh"

#ifdef G4UI_USE_XM
#include "G4UIXm.hh"
#endif

#include "QGSP_BERT.hh"
#include <iostream>

int main(int argc,char** argv)
{
    // Detect interactive mode (if no arguments) and define UI session
    //
    G4UIExecutive* ui = 0;
    char* filename = "out";
    if ( argc == 1 ) {
        ui = new G4UIExecutive(argc, argv);
        //filename = argv[1];
    } else if (argc == 3) {
        //char ** tmp = (char **)malloc(1 * sizeof(char *));
        //tmp[0]= (char *)malloc(50 * sizeof(char));
        //sprintf(tmp[0],argv[2]);
        //ui = new G4UIExecutive(argc-1,argv);
        filename = argv[2];
    } else if (argc ==2){
        ui = new G4UIExecutive(argc - 1, argv);
        filename = argv[1];
    }

    // Choose the Random engine
    G4Random::setTheEngine(new CLHEP::RanecuEngine);

    // Construct the default run manager
    muRunManager * runManager = new muRunManager;

    // Construct the analyzer
    muAnalyzer* analyzer = new muAnalyzer();
    analyzer->SetInit(false, filename);
    analyzer->Init();

    // Set mandatory initialization classes
    muDetectorConstruction* detector = new muDetectorConstruction();
    detector->SetVoxelFileName(filename);
    detector->SetAnalyzer(analyzer);
    runManager->SetUserInitialization(detector);

    // Physics list
    G4VModularPhysicsList* physicsList = new QGSP_BERT;
    physicsList->SetVerboseLevel(1);
    runManager->SetUserInitialization(physicsList);

    // Set user action classes
    //G4VUserPrimaryGeneratorAction* gen_action = new muPrimaryGeneratorAction(detector); // [yy] for gps
    G4VUserPrimaryGeneratorAction* gen_action = new muPrimaryGeneratorAction();
    runManager->SetUserAction(gen_action);

    muRunAction* run_action = new muRunAction;
    runManager->SetUserAction(run_action);

    muEventAction* event_action = new muEventAction(run_action);
    runManager->SetUserAction(event_action);

    //muSteppingAction* stepping_action = new muSteppingAction();
    //runManager->SetUserAction(stepping_action);

    //Initialize G4 kernel
    //runManager->Initialize();

    // Initialize visualization
    //
    G4VisManager* visManager = new G4VisExecutive;
    // G4VisExecutive can take a verbosity argument - see /vis/verbose guidance.
    // G4VisManager* visManager = new G4VisExecutive("Quiet");
    visManager->Initialize();

    // Get the pointer to the User Interface manager
    G4UImanager* UImanager = G4UImanager::GetUIpointer();

    // Process macro or start UI session
    //
    if ( ! ui ) {
        // batch mode
        G4String command = "/control/execute ";
        G4String fileName = argv[1];
        UImanager->ApplyCommand(command+fileName);
    }
    else {
        // interactive mode
        UImanager->ApplyCommand("/control/execute init_vis.mac");
        ui->SessionStart();
        delete ui;
    }

    analyzer->Terminate();

    // Job termination
    // Free the store: user actions, physics_list and detector_description are
    // owned and deleted by the run manager, so they should not be deleted
    // in the main() program !

    delete visManager;
    delete analyzer;
    delete runManager;
}
