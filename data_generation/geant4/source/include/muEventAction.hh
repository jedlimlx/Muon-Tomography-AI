
#ifndef muEventAction_h
#define muEventAction_h 1

#include "G4UserEventAction.hh"
#include "globals.hh"

class muRunAction;



class muEventAction : public G4UserEventAction
{
public:
    muEventAction(muRunAction*);
    ~muEventAction();
    
public:
    void  BeginOfEventAction(const G4Event*);
    void  EndOfEventAction(const G4Event*);
    
    
private:
    muRunAction* runAction;
    
};


#endif


