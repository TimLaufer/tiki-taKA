"""A tiny synthetic EPTS export (two track slots, five frames) in the layout of
the competition bundle, so the loader can be tested without the real data."""
import pytest

from src.data_loader import load_config

META = """<?xml version="1.0" encoding="utf-8"?>
<main>
  <Metadata>
    <GlobalConfig>
      <FrameRate>25</FrameRate>
      <ProviderName>Metrica Sports</ProviderName>
      <ProviderGlobalParameters>
        <ProviderParameter><Name>first_half_start</Name><Value>1</Value></ProviderParameter>
        <ProviderParameter><Name>first_half_end</Name><Value>3</Value></ProviderParameter>
        <ProviderParameter><Name>second_half_start</Name><Value>4</Value></ProviderParameter>
        <ProviderParameter><Name>second_half_end</Name><Value>5</Value></ProviderParameter>
      </ProviderGlobalParameters>
    </GlobalConfig>
    <Sessions>
      <Session id="s1">
        <MatchParameters><Score idLocalTeam="TH" idVisitingTeam="TA">
          <LocalTeamScore>0</LocalTeamScore><VisitingTeamScore>0</VisitingTeamScore>
        </Score></MatchParameters>
        <FieldSize><Width>105</Width><Height>68</Height></FieldSize>
      </Session>
    </Sessions>
    <Teams>
      <Team id="TH"><Name>Home FC</Name><ProviderTeamsParameters>
        <ProviderParameter><Name>attack_direction_first_half</Name><Value>left_to_right</Value></ProviderParameter>
      </ProviderTeamsParameters></Team>
      <Team id="TA"><Name>Away FC</Name><ProviderTeamsParameters/></Team>
    </Teams>
    <Players>
      <Player id="1" teamId="TH"><Name>Track 1</Name><ShirtNumber>1</ShirtNumber><ProviderPlayerParameters/></Player>
      <Player id="2" teamId="TA"><Name>Track 2</Name><ShirtNumber>2</ShirtNumber><ProviderPlayerParameters/></Player>
    </Players>
    <Devices><Device id="d1"><Sensors>
      <Sensor id="position"><Name>Position</Name><Channels>
        <Channel id="x"><Name>position_x</Name><Unit>normalized</Unit></Channel>
        <Channel id="y"><Name>position_y</Name><Unit>normalized</Unit></Channel>
      </Channels></Sensor>
      <Sensor id="speed"><Name>Speed in kilometers per hour</Name><Channels>
        <Channel id="s"><Name>speed_kph</Name><Unit>kph</Unit></Channel>
      </Channels></Sensor>
    </Sensors></Device></Devices>
    <PlayerChannels>
      <PlayerChannel id="p1_x" playerId="1" channelId="x"/>
      <PlayerChannel id="p1_y" playerId="1" channelId="y"/>
      <PlayerChannel id="p1_s" playerId="1" channelId="s"/>
      <PlayerChannel id="p2_x" playerId="2" channelId="x"/>
      <PlayerChannel id="p2_y" playerId="2" channelId="y"/>
      <PlayerChannel id="p2_s" playerId="2" channelId="s"/>
    </PlayerChannels>
  </Metadata>
  <DataFormatSpecifications>
    <DataFormatSpecification startFrame="1" endFrame="5" separator=":">
      <StringRegister name="frameCount"/>
      <SplitRegister separator=";">
        <SplitRegister separator=",">
          <PlayerChannelRef playerChannelId="p1_x"/><PlayerChannelRef playerChannelId="p1_y"/>
          <PlayerChannelRef playerChannelId="p1_s"/>
        </SplitRegister>
        <SplitRegister separator=",">
          <PlayerChannelRef playerChannelId="p2_x"/><PlayerChannelRef playerChannelId="p2_y"/>
          <PlayerChannelRef playerChannelId="p2_s"/>
        </SplitRegister>
      </SplitRegister>
      <SplitRegister separator=",">
        <BallChannelRef channelId="x"/><BallChannelRef channelId="y"/><BallChannelRef channelId="s"/>
      </SplitRegister>
    </DataFormatSpecification>
  </DataFormatSpecifications>
</main>
"""

# frame_id 1..5; frame 3 has no ball, frame 4 lost slot 2 and the ball speed,
# frame 5 has no line at all. Same layout as the real export: x,y,s per track, ball x,y,s.
RAW = """1:0.10,0.20,1.5;0.60,0.70,2.5:0.50,0.40,12.0
2:0.11,0.21,1.6;0.61,0.71,2.6:0.51,0.41,12.5
3:0.12,0.22,1.7;0.62,0.72,2.7:NaN,NaN,NaN
4:0.13,0.23,1.8;,,:0.53,0.43,
"""


@pytest.fixture
def epts_config(tmp_path, monkeypatch):
    monkeypatch.delenv("MTAH_DATA_DIR", raising=False)
    folder = tmp_path / "data" / "tactical"
    folder.mkdir(parents=True)
    (folder / "TEST_tactical_atd.xml").write_text(META)
    (folder / "TEST_tactical_atd.txt").write_text(RAW)
    config = load_config()
    config["data"] = {
        "dir": str(tmp_path / "data"),
        "cache_dir": str(tmp_path / "cache"),
        "game": "TEST",
        "source": "tactical",
    }
    config["phases"] = [
        {"name": "first_half", "first": 0, "last": 2, "half": 1, "graded": True, "clock_start_min": 0},
        {"name": "halftime", "first": 3, "last": 3, "half": 0, "graded": False},
        {"name": "second_half", "first": 4, "last": 4, "half": 2, "graded": True, "clock_start_min": 45},
    ]
    return config
